# %% [markdown]
# <a href="https://colab.research.google.com/github/gekoramy/uni.deep-learning/blob/finetune-like-you-pretrain/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# %%shell
tee requirements.txt << END
ftfy
jaxtyping
jupyter
matplotlib
pydantic
regex
torch
torchinfo
torchvision
tqdm
ultralytics
END

pip install -q -r requirements.txt
pip install -q git+https://github.com/openai/CLIP.git

# %%
import clip
import json
import os
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import PIL
import itertools as it
import math

from datetime import datetime
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchinfo import summary
from typing import Literal, Callable, Mapping, TypeVar
from tqdm import tqdm
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

# %%
device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
device


# %% [markdown]
# #### Utils

# %%
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# %%
# args:
#  - predictionList: [Prediction]
#  - numPred: int :: if numPred==-1 (default) consider all the predictions in predictionList
def display_predictions(predictionList, numPred=-1):
    limit = 0
    for p in predictionList:
        if numPred != -1 and limit >= numPred:
            return
        limit += 1

        p_image = p.image

        if not isinstance(p_image, torch.Tensor):
            p_image = torchvision.transforms.PILToTensor()(p_image)

        p_description = p.description
        p_ground_truth_bbox = p.ground_truth_bbox
        p_output_bbox = p.output_bbox

        # TODO: concatenate
        p_image = draw_bounding_boxes(
            p_image, p_ground_truth_bbox.unsqueeze(0), colors="green", width=5
        )
        p_image = draw_bounding_boxes(
            p_image, p_output_bbox.unsqueeze(0), colors="red", width=5
        )

        tensor_to_pil = transforms.ToPILImage()
        image_pil = tensor_to_pil(p_image)
        display(image_pil)
        print(p_description)
        print("\n\n")


# %% [markdown]
# #### Dataset and type declaration

# %%
# %%shell
if ! [ -d dataset ]; then
  mkdir dataset &&
  gdown 1P8a1g76lDJ8cMIXjNDdboaRR5-HsVmUb &&
  tar -xf refcocog.tar.gz -C dataset &&
  rm refcocog.tar.gz
fi

# %%
root = os.path.join("dataset", "refcocog", "")
data_instances = os.path.join(root, "annotations", "instances.json")
data_refs = os.path.join(root, "annotations", "refs(umd).p")
data_images = os.path.join(root, "images", "")

# %%
I = TypeVar("I")
P = TypeVar("P")
B = TypeVar("B")
T = TypeVar("T")

Img = UInt[torch.Tensor, "C W H"]
BBox = UInt[torch.Tensor, "4"]
Split = Literal["train", "test", "val"]


@dataclass
class Info:
    description: str  # This is stable 1.0 version of the 2014 MS COCO dataset.
    url: str  # http://mscoco.org/
    version: str  # 1.0
    year: int  # 2014
    contributor: str  # Microsoft COCO group
    date_created: datetime  # 2015-01-27 09:11:52.357475


@dataclass
class Image:
    license: int  # each image has an associated licence id
    file_name: str  # file name of the image
    coco_url: str  # example http://mscoco.org/images/131074
    height: int
    width: int
    flickr_url: str  # example http://farm9.staticflickr.com/8308/7908210548_33e
    id: int  # id of the imag
    date_captured: datetime  # example '2013-11-21 01:03:06'


@dataclass
class License:
    url: str  # example http://creativecommons.org/licenses/by-nc-sa/2.0/
    id: int  # id of the licence
    name: str  # example 'Attribution-NonCommercial-ShareAlike License


@dataclass
class Annotation:
    # segmentation: list[list[float]]  # description of the mask; example [[44.17, 217.83, 36.21, 219.37, 33.64, 214.49, 31.08, 204.74, 36.47, 202.68, 44.17, 203.2]]
    area: float  # number of pixel of the described object
    iscrowd: Literal[
        1, 0
    ]  # Crowd annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people)
    image_id: int  # id of the target image
    bbox: tuple[
        float, float, float, float
    ]  # bounding box coordinates [xmin, ymin, width, height]
    category_id: int
    id: int  # annotation id


@dataclass
class Category:
    supercategory: str  # example 'vehicle'
    id: int  # category id
    name: str  # example 'airplane'


@dataclass
class Instances:
    info: Info
    images: list[Image]
    licenses: list[License]
    annotations: list[Annotation]
    categories: list[Category]


@dataclass
class Sentence:
    tokens: list[str]  # tokenized version of referring expression
    raw: str  # unprocessed referring expression
    sent: str  # referring expression with mild processing, lower case, spell correction, etc.
    sent_id: int  # unique referring expression id


@dataclass
class Ref:
    image_id: int  # unique image id
    split: Split
    sentences: list[Sentence]
    file_name: str  # file name of image relative to img_root
    category_id: int  # object category label
    ann_id: int  # id of object annotation in instance.json
    sent_ids: list[int]  # same ids as nested sentences[...][sent_id]
    ref_id: int  # unique id for refering expression


# %%
class Prediction:
    def __init__(self, image, description, ground_truth_bbox, output_bbox):
        self.image = image
        self.description = description
        self.ground_truth_bbox = ground_truth_bbox
        self.output_bbox = output_bbox


# %%
def fix_ref(x: Ref) -> Ref:
    x.file_name = fix_filename(x.file_name)
    return x


def fix_filename(x: str) -> str:
    """
    :param x: COCO_..._[image_id]_[annotation_id].jpg
    :return:  COCO_..._[image_id].jpg

    >>> fix_filename('COCO_..._[image_id]_0000000001.jpg')
    'COCO_..._[image_id].jpg'

    """
    return re.sub("_\d+\.jpg$", ".jpg", x)


# %%
with open(data_refs, "rb") as f:
    raw = pickle.load(f)

refs: list[Ref] = [fix_ref(Ref(**ref)) for ref in raw]

# %%
with open(data_instances, "r") as f:
    raw = json.load(f)

instances: Instances = Instances(**raw)

id2annotation: Mapping[int, Annotation] = {x.id: x for x in instances.annotations}


# %%
class CocoDataset(Dataset[tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]]):
    def __init__(
        self,
        split: Split,
        limit: int = -1,
    ):
        self.__init__
        self.items: list[tuple[str, list[str], Float[torch.Tensor, "4"]]] = [
            (i, [s.sent for s in ss], xywh)
            for ref in refs
            if ref.split == split
            for i in [os.path.join(data_images, ref.file_name)]
            for ss in [ref.sentences]
            for xywh in [
                torch.tensor(id2annotation[ref.ann_id].bbox, dtype=torch.float)
            ]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(
        self, index: int
    ) -> tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]:
        i, ps, xywh = self.items[index]
        xyxy: Float[torch.Tensor, "4"] = torchvision.ops.box_convert(
            xywh, in_fmt="xywh", out_fmt="xyxy"
        )
        with PIL.Image.open(i) as img:
            img.load()
            return img, ps, xyxy


# %%
class Coco4CLIPDataset(Dataset[tuple[list[PIL.Image], list[str]]]):
    def __init__(
        self,
        split: Split,
        limit: int = -1,
    ):
        self.__init__
        self.items: list[tuple[str, list[str], Float[torch.Tensor, "4"]]] = [
            (i, [s.sent for s in ss], xywh)
            for ref in refs
            if ref.split == split
            for i in [os.path.join(data_images, ref.file_name)]
            for ss in [ref.sentences]
            for xywh in [
                torch.tensor(id2annotation[ref.ann_id].bbox, dtype=torch.float)
            ]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[list[PIL.Image], list[str]]:
        i, ps, xywh = self.items[index]
        xyxy: Float[torch.Tensor, "4"] = torchvision.ops.box_convert(
            xywh, in_fmt="xywh", out_fmt="xyxy"
        )
        with PIL.Image.open(i) as img:
            img.load()
            return [img.crop(xyxy.tolist())], ps


# %%
def unzip(batch: list[tuple[T, ...]]) -> tuple[list[T], ...]:
    return tuple(zip(*batch))


# %%
batch_size: int = 3
limit: int = 5 * batch_size

# %%
dl: DataLoader[
    tuple[list[PIL.Image], list[list[str]], list[Float[torch.Tensor, "4"]]]
] = DataLoader(
    dataset=CocoDataset(split="test", limit=limit),
    batch_size=batch_size,
    collate_fn=unzip,
)

# %%
import os

dl4clip: DataLoader[tuple[list[PIL.Image], list[str]]] = DataLoader(
    dataset=Coco4CLIPDataset(split="test", limit=limit),
    batch_size=batch_size,
    collate_fn=unzip,
    generator=torch.Generator(device=device),  # add for GPU
    shuffle=True,
)

# %%
imgs: tuple[PIL.Image, ...]
promptss: tuple[list[str], ...]
true_xyxy: tuple[Float[torch.Tensor, "4"], ...]

for imgs, promptss, true_xyxy in dl:
    print(imgs)
    print(promptss)
    print(true_xyxy)
    print("-" * 50)

# %%
cropss: tuple[list[PIL.Image], ...]
promptss: tuple[list[str], ...]

for cropss, promptss in dl4clip:
    print(cropss)
    print(promptss)
    print("-" * 50)


# %% [markdown]
# # Fine tune like you pretrain
# In the following we try to fine tune CLIP image and text encoders using contrastive learning as proposed by the original paper.

# %%
class FLYP_CLIP(nn.Module):
    def __init__(
        self, device=device
    ):  # TODO: aggiungere device=device anche nelle architetture dello standard fine tuning
        super().__init__()

        model, preprocess = clip.load("RN50")

        # freeze all pretrained layers by setting requires_grad=False
        for param in model.parameters():
            param.requires_grad = False

        self.clip_visual_encoder = model.encode_image
        self.clip_text_encoder = model.encode_text
        self.clip_visual_preprocess = preprocess
        self.clip_text_preprocess = clip.tokenize

        self.visual_encoder_linearHead = nn.Linear(1024, 1024)
        self.text_encoder_linearHead = nn.Linear(1024, 1024)

        # the temperature parameter is added as suggested by the original paper in order to prevent training instability
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # preprocess input prompts as required by the visual encoder
    def visual_preprocess(self, _imgs):
        prep_images = torch.stack([self.clip_visual_preprocess(i) for i in _imgs]).to(
            device
        )

        return prep_images

    # preprocess text prompts as required by the text encoder
    def text_preprocess(self, _txts):
        prep_texts = self.clip_text_preprocess(_txts)

        return prep_texts

    # visual encoder
    def visual_encoder(self, image):
        with torch.no_grad():
            clipFeatures = self.clip_visual_encoder(image).float()  # add for GPU

        x = F.relu(clipFeatures)
        x = self.visual_encoder_linearHead(x)

        return x

    # text encoder
    def text_encoder(self, text):
        with torch.no_grad():
            clipFeatures = self.clip_text_encoder(text).float()  # add for GPU

        x = F.relu(clipFeatures)
        x = self.text_encoder_linearHead(x)

        return x

    def forward(self, image, text):
        with torch.no_grad():
            image_pre = self.visual_preprocess(image)
            text_pre = self.text_preprocess(text)

        image_features = self.visual_encoder(image_pre)
        text_features = self.text_encoder(text_pre)

        return image_features, text_features, self.logit_scale.exp()


# %%
def get_optimizer(model, _lr, _wd, _momentum):
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=_lr, weight_decay=_wd, momentum=_momentum
    )
    return optimizer


# %%
class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, num_logits):
        labels = torch.arange(num_logits)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale):
        # compute logits per image and logits per text
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        # get ground truth labels for the computation of the cross entropy loss
        labels = self.get_ground_truth(logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss


# %%
def training_step(
    model: torch.nn.Module,  # neural network to be trained
    data_loader: torch.utils.data.DataLoader,  # data loader to be iterated
    loss_fn: torch.nn.Module,  # loss function
    optimizer: torch.optim.Optimizer,  # optimizer
    device: torch.device = device,  # target device
):
    train_loss = 0.0

    model.to(device)
    model.train()

    for batch_idx, (cropss, promptss) in tqdm(enumerate(data_loader)):
        # for this implementation we consider only one prompt for each crop
        model_input_crops = [c[0] for c in cropss]
        model_input_prompts = [p[0] for p in promptss]

        # send data to target device
        ####cropss = cropss.to(device)
        ####promptss = promptss.to(device)

        # forward computation
        model_out = model(model_input_crops, model_input_prompts)
        image_features = model_out[0]
        text_features = model_out[1]
        logit_scale = model_out[2]

        # calculate loss
        loss = loss_fn(image_features, text_features, logit_scale)
        train_loss += loss

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))

    # Calculate loss per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"\nTrain loss: {train_loss:.5f}\n")
    return train_loss


# %%
def test_step(
    model: torch.nn.Module,  # neural network to be evaluated
    data_loader: torch.utils.data.DataLoader,  # data loader to be iterated
    loss_fn: torch.nn.Module,  # loss function
    device: torch.device = device,  # target device
):
    test_loss = 0.0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        # for batch_idx, cropss, promptss in tqdm(enumerate(data_loader)):
        for batch_idx, (cropss, promptss) in tqdm(enumerate(data_loader)):
            # for this implementation we consider only one prompt for each crop
            model_input_crops = [c[0] for c in cropss]
            model_input_prompts = [p[0] for p in promptss]

            # send data to target device
            ####cropss = cropss.to(device)
            ####promptss = promptss.to(device)

            # forward computation
            model_out = model(model_input_crops, model_input_prompts)
            image_features = model_out[0]
            text_features = model_out[1]
            logit_scale = model_out[2]

            # calculate loss
            loss = loss_fn(image_features, text_features, logit_scale)
            test_loss += loss

        test_loss /= len(data_loader)
        print(f"\nTest loss: {test_loss:.5f}\n")
        return test_loss


# %%
# instantiate the network and move it to the chosen device
net = FLYP_CLIP().to(device)


# %%
# tensorboard logging utilities
def log_values(writer, step, loss, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)


# %%
# setting a manual seed allow us to provide reprudicible results in this notebook
torch.manual_seed(42)

# create a logger for the experiment
writer = SummaryWriter(log_dir="runs/exp1")

BATCH_SIZE = 3
LIMIT = 5 * BATCH_SIZE
NUM_WORKERS = 1

# get dataset instance
train_dataset = Coco4CLIPDataset(split="train", limit=LIMIT)
test_dataset = Coco4CLIPDataset(split="test", limit=LIMIT)
val_dataset = Coco4CLIPDataset(split="val", limit=LIMIT)
print(
    f"LEN_TRAIN_DATASET: {len(train_dataset)}, LEN_TEST_DATASET: {len(test_dataset)}, LEN_VALIDATION_DATASET: {len(val_dataset)}"
)

# get dataloaders
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
train_loader: DataLoader[tuple[list[PIL.Image], list[str]]] = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=unzip,
    generator=torch.Generator(device=device),  # add for GPU
    shuffle=True,
)
test_loader: DataLoader[tuple[list[PIL.Image], list[str]]] = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=unzip,
    shuffle=False,
)
val_loader: DataLoader[tuple[list[PIL.Image], list[str]]] = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=unzip,
    shuffle=False,
)
print(
    f"LEN_TRAIN_DATALOADER: {len(train_loader)}, LEN_TEST_DATALOADER: {len(val_loader)}, LEN_VALIDATION_DATALOADER: {len(test_loader)}"
)

# instantiate the optimizer
learning_rate = 0.01
weight_decay = 0.000001
momentum = 0.9
optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

# define the cost function
cost_function = ClipLoss().to(device)

print("Before training:")
train_loss = test_step(model=net, data_loader=train_loader, loss_fn=cost_function)
val_loss = test_step(model=net, data_loader=val_loader, loss_fn=cost_function)
test_loss = test_step(model=net, data_loader=test_loader, loss_fn=cost_function)

# log to TensorBoard
log_values(writer, -1, train_loss, "train")
log_values(writer, -1, val_loss, "validation")
log_values(writer, -1, test_loss, "test")

print("\tTraining loss {:.5f}".format(train_loss))
print("\tValidation loss {:.5f}".format(val_loss))
print("\tTest loss {:.5f}".format(test_loss))
print("-----------------------------------------------------")

# measure time
train_time_start = timer()

EPOCHS = 3
for epoch in tqdm(range(EPOCHS)):
    train_loss = training_step(
        model=net, data_loader=train_loader, loss_fn=cost_function, optimizer=optimizer
    )

    val_loss = test_step(model=net, data_loader=val_loader, loss_fn=cost_function)

    # logs to TensorBoard
    log_values(writer, epoch, train_loss, "train")
    log_values(writer, epoch, val_loss, "validation")

    print("Epoch: {:d}".format(epoch + 1))
    print("\tTraining loss {:.5f}".format(train_loss))
    print("\tValidation loss {:.5f}".format(val_loss))
    print("-----------------------------------------------------")

train_time_end = timer()
total_train_time_model_1 = print_train_time(
    start=train_time_start, end=train_time_end, device=device
)
# compute final evaluation results
print("After training:")
train_loss = test_step(model=net, data_loader=train_loader, loss_fn=cost_function)
val_loss = test_step(model=net, data_loader=val_loader, loss_fn=cost_function)
test_loss = test_step(model=net, data_loader=test_loader, loss_fn=cost_function)

# log to TensorBoard
log_values(writer, EPOCHS, train_loss, "train")
log_values(writer, EPOCHS, val_loss, "validation")
log_values(writer, EPOCHS, test_loss, "test")

print("\tTraining loss {:.5f}".format(train_loss))
print("\tValidation loss {:.5f}".format(val_loss))
print("\tTest loss {:.5f}".format(test_loss))
print("-----------------------------------------------------")

# closes the logger
writer.close()


# %% [markdown]
# # Test the model on our down stream task
# In the following of the notebook we test the performance of the trained model on our objective task.

# %% [markdown]
# ## Yolov5

# %%
class Yolo_v5(torch.nn.Module):
    def __init__(self, device=device):
        super().__init__()

        # load yolo model
        self.yolo_model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True
        )
        self.yolo_model.to(device=device).eval()

    def forward(self, img):
        # yolo bboxes
        predictions = self.yolo_model(img)

        # xmin,      ymin,      xmax,      ymax,      confidence, class
        # 274.06390, 231.20389, 392.66345, 372.59018, 0.93251,    23.00000
        bboxes: list[
            Float[torch.Tensor, "X 6"]
        ] = (
            predictions.xyxy
        )  # bboxes[i] contains the bboxes highlighted by yolo in image i

        for image_idx, bbox_img in enumerate(bboxes):
            # if empty, put a bbox equal to image size
            if len(bbox_img) == 0:
                bboxes[image_idx] = torch.tensor(
                    [[0, 0, img[image_idx].size[0], img[image_idx].size[1], 0, 0]],
                    dtype=torch.float,
                )

        return bboxes


# %%
# instantiate the region proposal algorithm
yolo = Yolo_v5().to(device)


# %% [markdown]
# ## Evaluation code

# %%
def get_accuracy_function():
    def iou_accuracy(bbox_prediction, bbox_groundtruth):
        # compute intersection over union between ground truth bboxes and predicted bboxes
        iou_accuracy_matrix = torchvision.ops.box_iou(
            bbox_prediction[:, :4], bbox_groundtruth
        )

        # extract the diagonal elements
        iou_accuracy_matrix_diagonal = torch.diag(iou_accuracy_matrix)

        # compute the mean of the intersection over union
        mean_iou = iou_accuracy_matrix_diagonal.mean()

        # compute the iou accuracy
        iou_accuracy_output = mean_iou.item()

        return iou_accuracy_output

    return iou_accuracy


# %%
# input:
#   -> retrived_bboxes : bounding boxes proposed by the region proposal model
#   -> bbox_groundtruth : ground truth bounding box provided by the training sample
# output:
#   -> [3, 5] in this case for the first element in the batch the best bbox is the fourth, while for the second element in the batch the best bbox is the sixth. The best bbox is the one characterized by the largest IoU with the ground truth bbox
def best_bbox_one_hot_encoding(retrived_bboxes, bbox_groundtruth):
    batch_bbox_one_hot_encoding = []
    for batch_item_retrived_bboxes, batch_item_bbox_groundtruth in zip(
        retrived_bboxes, bbox_groundtruth
    ):
        iou_matrix = torchvision.ops.box_iou(
            batch_item_retrived_bboxes[:, :4], batch_item_bbox_groundtruth.unsqueeze(0)
        )
        batch_bbox_one_hot_encoding.append(torch.argmax(iou_matrix, dim=0))

    batch_bbox_one_hot_encoding = torch.cat(batch_bbox_one_hot_encoding, dim=0)

    return batch_bbox_one_hot_encoding


# %%
def evaluation(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    region_proposal_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, iou_test_acc = 0, 0
    model.to(device)
    region_proposal_model.to(device)

    with torch.inference_mode():
        for batch_idx, (imgs, promptss, true_xyxy) in tqdm(enumerate(data_loader)):
            # send data to target device
            # todo: send data to target device

            # forward pass
            # i. region proposal
            bboxes = region_proposal_model(imgs)

            # ii. get best bounding box with respect to the ground truth
            bbox_groundtruth = best_bbox_one_hot_encoding(bboxes, true_xyxy)

            # from yolo bboxes to cropped images
            crops = []
            for batch_image, batch_image_bboxes in zip(imgs, bboxes):
                list_bboxes_image: list[Image] = [
                    batch_image.crop((xmin, ymin, xmax, ymax))
                    for bbox in batch_image_bboxes
                    for [xmin, ymin, xmax, ymax, _, _] in [bbox.tolist()]
                ]

                crops.append(list_bboxes_image)

            # forward pass
            cropss_z = []
            promptss_z = []
            for c, p in zip(crops, promptss):
                model_output = model(c, p)
                model_output_image_features = model_output[0]
                model_output_text_features = model_output[1]
                _ = model_output[2]

                cropss_z.append(model_output_image_features)
                promptss_z.append(model_output_text_features)

            # cosine similarity evaluation
            #   cropss_z :: list of BATCH_SIZE tensors: [tensor([bbox_img_1, 1024]), tensor([bbox_img_2, 1024]), ..., tensor([bbox_img_BATCH_SIZE, 1024])]
            #   promptss_z :: list of BATCH_SIZE tensors: [tensor([prompts_img_1, 1024]), tensor([prompts_img_2, 1024]), ..., tensor([prompts_img_BATCH_SIZE, 1024])]
            bbox_index_pred = (
                []
            )  # for each batch sample this list contains the index of the predicted bbox at the end of the iteration
            loss = 0.0
            for c_z, p_z, y in zip(cropss_z, promptss_z, bbox_groundtruth):
                crop_logits = (
                    []
                )  # for each crop we set the average cosine similarity with the prompts
                for vector_c_z in c_z:
                    vector_c_z_cos_similarities = []
                    for vector_p_z in p_z:
                        cosine_similarity = torch.nn.CosineSimilarity()(
                            vector_c_z.unsqueeze(0), vector_p_z.unsqueeze(0)
                        ).item()
                        vector_c_z_cos_similarities.append(cosine_similarity)

                    mean_cosine_similarity = sum(vector_c_z_cos_similarities) / len(
                        vector_c_z_cos_similarities
                    )

                    crop_logits.append(mean_cosine_similarity)

                # calculate loss
                loss = loss + loss_fn(
                    torch.tensor(crop_logits).to(device), y.to(device)
                )

                # get index of the predicted bounding box in order to compute IoU accuracy
                bbox_index_pred.append(crop_logits.index(max(crop_logits)))

            loss = loss / len(bbox_groundtruth)  # avg loss
            test_loss += loss

            # get predicted bounding box for each example in the batch
            bbox_pred = [
                batch_example_bboxes[idx]
                for batch_example_bboxes, idx in zip(bboxes, bbox_index_pred)
            ]

            prediction_obj = Prediction(
                imgs[0], promptss[0], true_xyxy[0], bbox_pred[0][:4]
            )
            display_predictions([prediction_obj])

            # calculate intersection over union train accuracy
            acc = accuracy_fn(
                torch.stack(bbox_pred, dim=0), torch.stack(list(true_xyxy), dim=0)
            )
            iou_test_acc += acc

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        iou_test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | IoU test accuracy: {iou_test_acc:.5f}\n")
        return test_loss, iou_test_acc


# %%
# tensorboard logging utilities
def log_values_evaluation(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


# %%
# setting a manual seed allow us to provide reprudicible results in this notebook
torch.manual_seed(42)

# create a logger for the experiment
writer = SummaryWriter(log_dir="runs/exp1")

BATCH_SIZE = 3
LIMIT = 5 * BATCH_SIZE

# get dataset instance
train_dataset = CocoDataset(split="train", limit=LIMIT)
test_dataset = CocoDataset(split="test", limit=LIMIT)
val_dataset = CocoDataset(split="val", limit=LIMIT)
print(
    f"LEN_TRAIN_DATASET: {len(train_dataset)}, LEN_TEST_DATASET: {len(test_dataset)}, LEN_VALIDATION_DATASET: {len(val_dataset)}"
)

# get dataloaders
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
train_loader: DataLoader[
    tuple[list[PIL.Image], list[list[str]], list[Float[torch.Tensor, "4"]]]
] = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=unzip,
)
test_loader: DataLoader[
    tuple[list[PIL.Image], list[list[str]], list[Float[torch.Tensor, "4"]]]
] = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=unzip,
)
val_loader: DataLoader[
    tuple[list[PIL.Image], list[list[str]], list[Float[torch.Tensor, "4"]]]
] = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=unzip,
)
print(
    f"LEN_TRAIN_DATALOADER: {len(train_loader)}, LEN_TEST_DATALOADER: {len(val_loader)}, LEN_VALIDATION_DATALOADER: {len(test_loader)}"
)

# define the cost function
loss_function = torch.nn.CrossEntropyLoss()

# define the accuracy function
accuracy_fn = get_accuracy_function()

print("Evalutation on the downstream task:")
train_loss, train_accuracy = evaluation(
    data_loader=train_loader,
    model=net,
    region_proposal_model=yolo,
    loss_fn=loss_function,
    accuracy_fn=accuracy_fn,
)
test_loss, test_accuracy = evaluation(
    data_loader=test_loader,
    model=net,
    region_proposal_model=yolo,
    loss_fn=loss_function,
    accuracy_fn=accuracy_fn,
)
val_loss, val_accuracy = evaluation(
    data_loader=val_loader,
    model=net,
    region_proposal_model=yolo,
    loss_fn=loss_function,
    accuracy_fn=accuracy_fn,
)

# log to TensorBoard
log_values_evaluation(writer, -1, train_loss, train_accuracy, "train")
log_values_evaluation(writer, -1, val_loss, val_accuracy, "validation")
log_values_evaluation(writer, -1, test_loss, test_accuracy, "test")

print(
    "\tTraining loss {:.5f}, Training accuracy {:.5f}".format(
        train_loss, train_accuracy
    )
)
print(
    "\tValidation loss {:.5f}, Validation accuracy {:.5f}".format(
        val_loss, val_accuracy
    )
)
print("\tTest loss {:.5f}, Test accuracy {:.5f}".format(test_loss, test_accuracy))
print("-----------------------------------------------------")

# closes the logger
writer.close()
