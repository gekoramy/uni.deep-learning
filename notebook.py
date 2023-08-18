# %% [markdown]
# # Standard finetuning
# In this notebook we propose the straightforward solution to fine tune CLIP. The general idea is to add linear layer(s) on top of the 1024 visual features of CLIP.

# %% [markdown]
# Dependences

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
torchvision
torchinfo
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

from datetime import datetime
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchinfo import summary
from typing import Literal, Callable, Mapping, TypeVar, Any, Optional
from tqdm import tqdm
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

# %%
device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
device

# %% [markdown]
# ## Load the dataset
# First of all we have to load the dataset.

# %%
# %%shell
if ! [ -d dataset ]; then
  mkdir dataset &&
  gdown 1P8a1g76lDJ8cMIXjNDdboaRR5-HsVmUb &&
  tar -xf refcocog.tar.gz -C dataset &&
  rm refcocog.tar.gz
fi

# %% [markdown]
# Folder paths

# %%
root = os.path.join("dataset", "refcocog", "")
data_instances = os.path.join(root, "annotations", "instances.json")
data_refs = os.path.join(root, "annotations", "refs(umd).p")
data_images = os.path.join(root, "images", "")

# %% [markdown]
# Type declaration

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
@dataclass
class Prediction:
    image: Any
    description: list[str]  # natural language descriptions of the area of interest
    ground_truth_bbox: tuple[float, float, float, float]  # ground truth bounding box
    output_bbox: tuple[float, float, float, float]  # predicted bounding box


# %% [markdown]
# Read the dataset infos

# %%
def fix_ref(x: Ref) -> Ref:
    x.file_name = fix_filename(x.file_name)
    return x


def fix_filename(x: str) -> str:
    """
    :param x: COCO_..._[image_id]_[annotation_id].jpg
    :return:  COCO_..._[image_id].jpg
    """
    return re.sub("_\d+\.jpg$", ".jpg", x)


# %%
with open(data_refs, "rb") as f:
    raw = pickle.load(f)

# %%
refs: list[Ref] = [fix_ref(Ref(**ref)) for ref in raw]

# %%
with open(data_instances, "r") as f:
    raw = json.load(f)

# %%
instances: Instances = Instances(**raw)

# %%
id2annotation: Mapping[int, Annotation] = {x.id: x for x in instances.annotations}


# %% [markdown]
# Define custom dataset

# %%
class CocoDataset(Dataset[tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]]):
    def __init__(
        self,
        split: Split,
        limit: int = -1,
    ):
        self.__init__
        self.items: list[tuple[str, list[str], Float[torch.Tensor, "4"]]] = [
            (i, [s.sent for s in ss], b)
            for ref in refs
            if ref.split == split
            for i in [os.path.join(data_images, ref.file_name)]
            for ss in [ref.sentences]
            for b in [torch.tensor(id2annotation[ref.ann_id].bbox, dtype=torch.float)]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(
        self, index: int
    ) -> tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]:
        i, ps, b = self.items[index]
        with PIL.Image.open(i) as img:
            img.load()
            return img, ps, b


# %% [markdown]
# ## Training free CLIP results
# For the sake of comparison with the implementations below, we have to evaluate CLIP training free with the same portion of the dataset.

# %% [markdown]
# Load yolo model

# %%
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, device=device)

# %% [markdown]
# Load CLIP model

# %%
clip_model, preprocess = clip.load("RN50", device=device)

# %% [markdown]
# Baseline evaluation

# %%
test_dataset: Dataset[tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]] = CocoDataset(
    split="test",
    limit=10
)

# %%
BATCH_SIZE: Optional[int] = None
NUM_WORKERS: int = os.cpu_count()

print(f"Creating DataLoader w/ batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

# %%
stored_predictions: list[Prediction] = []

ious: list[float] = []
coss: list[float] = []
euds: list[float] = []

batch: tuple[
    UInt[torch.Tensor, "1 C W H"], tuple[list[tuple[str]]], UInt[torch.Tensor, "1 4"]
]

with torch.no_grad():
    for img_pil, prompts, true_xywh in tqdm(test_dataloader):
        [true_xyxy] = torchvision.ops.box_convert(
            true_xywh.unsqueeze(0), in_fmt="xywh", out_fmt="xyxy"
        )

        # yolo bboxes
        predictions = yolo_model(img_pil)

        # xmin,      ymin,      xmax,      ymax,      confidence, class
        # 274.06390, 231.20389, 392.66345, 372.59018, 0.93251,    23.00000
        bboxes: Float[torch.Tensor, "X 6"] = predictions.xyxy[0]

        # if empty, put a bbox equal to image size
        if len(bboxes) == 0:
            bboxes = torch.tensor(
                [[0, 0, img_pil.size[0], img_pil.size[1], 0, 0]], dtype=torch.float
            )

        # from yolo bboxes to cropped images
        crops: list[Image] = [
            img_pil.crop((xmin, ymin, xmax, ymax))
            for bbox in bboxes
            for [xmin, ymin, xmax, ymax, _, _] in [bbox.tolist()]
        ]

        # clip preprocess on cropped images
        preprocess_crops: Float[torch.Tensor, "X 3 244 244"] = torch.stack(
            [preprocess(crop) for crop in crops]
        ).to(device=device)

        # format each available prompt
        prompts_tokens: Int[torch.Tensor, "P 77"] = clip.tokenize(
            [
                template.format(prompt)
                for template in ["{}", "A photo of {}", "We can see {}"]
                for prompt in prompts
            ]
        )

        # clip scores
        ass_z: tuple[
            Float[torch.Tensor, "X P"], Float[torch.Tensor, "P X"]
        ] = clip_model(preprocess_crops, prompts_tokens)
        _, logits_per_prompt = ass_z

        # final prediction
        best_match: int = torch.argmax(torch.max(logits_per_prompt, 0).values).item()
        prediction_bbox: Float[torch.Tensor, "4"] = bboxes[best_match][:4]

        # metrics
        iou: float = torchvision.ops.box_iou(
            true_xyxy.unsqueeze(0), prediction_bbox.unsqueeze(0)
        ).item()
        ious.append(iou)

        rectangle: tuple[int, int, int, int] = true_xyxy.tolist()
        ground_truth_crop: PIL.Image = img_pil.crop(rectangle)

        rectangle: tuple[int, int, int, int] = torch.tensor(
            prediction_bbox, dtype=torch.int
        ).tolist()
        prediction_crop: PIL.Image = img_pil.crop(rectangle)

        # from float16 to float32
        X: Float[torch.Tensor, "1"] = torch.tensor(
            clip_model.encode_image(
                torch.tensor(preprocess(ground_truth_crop)).unsqueeze(0)
            ),
            dtype=torch.float,
        )
        Y: Float[torch.Tensor, "1"] = torch.tensor(
            clip_model.encode_image(
                torch.tensor(preprocess(prediction_crop)).unsqueeze(0)
            ),
            dtype=torch.float,
        )

        cos: float = F.cosine_similarity(X, Y).item()
        coss.append(cos)

        eud: float = torch.cdist(X, Y, p=2).item()
        euds.append(eud)

        # store the prediction
        pred: Prediction = Prediction(
            image=img_pil,
            description=prompts,
            ground_truth_bbox=true_xyxy.tolist(),
            output_bbox=prediction_bbox.tolist(),
        )
        stored_predictions.append(pred)

        torch.cuda.empty_cache()

# %%
prompts_tokens.shape

# %%
preprocess_crops.shape

# %%
len(stored_predictions)

# %% [markdown]
# Performance:

# %%
print(f"ious: {torch.mean(torch.tensor(ious, dtype=torch.float))}")
print(f"coss: {torch.mean(torch.tensor(coss, dtype=torch.float))}")
print(f"euds: {torch.mean(torch.tensor(euds, dtype=torch.float))}")


# %% [markdown]
# Function to display a random sample of predictions.

# %%
# args:
#  - predictionList: [Prediction]
#  - numPred: int :: if numPred==-1 (default) consider all the predictions in predictionList
def display_predictions(predictionList: list[Prediction], numPred: int = -1):
    limit: int = len(predictionList) if numPred < 0 else min(numPred, len(predictionList))
    p: Prediction

    for p in predictionList[:limit]:
        img: PIL.Image = p.image.copy()
        img_d: PIL.ImageDraw = PIL.ImageDraw.Draw(img)
        img_d.rectangle(p.ground_truth_bbox, outline="green", width=5)
        img_d.rectangle(p.output_bbox, outline="red", width=5)

        display(img)
        print(p.description, "\n\n")


# %%
display_predictions(stored_predictions, 3)


# %% [markdown]
# ## Region proposal networks

# %% [markdown]
# ### YOLOv5

# %%
class Yolo_v5(torch.nn.Module):
    def __init__(self, device=device):
        super().__init__()

        # load yolo model
        yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", device=device, pretrained=True)

    def forward(
        self, img_pil: list[PIL.Image]
    ) -> list[Float[torch.Tensor, "X 6"]]:
        #######print(f"img_pil 0: {type(img_pil)}")
        #######print(f"img_pil 1: {len(img_pil)}")
        #######print("img_pil[0]")
        #######print(img_pil[0])
        #######print("img_pil[0].size")
        #######print(img_pil[0].size)

        # yolo bboxes
        predictions = yolo_model(img_pil)
        #######print(f"predictions type: {type(predictions)}")
        #######predictions.show()

        #######print("predictions.xyxy")
        #######print(predictions.xyxy)
        #######print("predictions.xyxy len")
        #######print(len(predictions.xyxy))

        # xmin,      ymin,      xmax,      ymax,      confidence, class
        # 274.06390, 231.20389, 392.66345, 372.59018, 0.93251,    23.00000
        bboxes: list[
            Float[torch.Tensor, "X 6"]
        ] = (
            predictions.xyxy
        )  # bboxes[i] contains the bboxes highlighted by yolo in image i

        #######print("len bboxes")
        #######print(len(bboxes))

        for image_idx, bbox_img in enumerate(bboxes):
            # if empty, put a bbox equal to image size
            if len(bbox_img) == 0:
                bboxes[image_idx] = torch.tensor(
                    [
                        [
                            0,
                            0,
                            img_pil[image_idx].size[0],
                            img_pil[image_idx].size[1],
                            0,
                            0,
                        ]
                    ],
                    dtype=torch.float,
                )  # TODO: test this piece of code

        return bboxes


# %% [markdown]
# ## Linear layers on top of image encoder

# %%
summary(clip_model)

# %%
# summary(clip_model.visual, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])

# %%
summary(
    clip_model.transformer,
    input_size=(77, 512),
    col_names=["input_size", "output_size", "num_params", "trainable"],
)


# %% [markdown]
# In the following cell we create a neural network that builds a linear head on top of the visual encoder of CLIP.

# %%
class CLIP_SF_image_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model, _ = clip.load("RN50")

        # take the visual encoder of CLIP
        # we also convert it to be 32 bit (by default CLIP is 16)
        self.encoder = model.visual

        # freeze all pretrained layers by setting requires_grad=False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # add a linear layer
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # visual encoder
        with torch.no_grad():
            x = self.encoder(x)
        # ---

        # linear head
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # ---

        return x


# %%
# test code to be deleted
"""
test_model = CLIP_SF_image_encoder()
test_model(preprocess(img_pil).unsqueeze(0))
"""


# %%
class CLIP_SF_text_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model, _ = clip.load("RN50")

        # take the text encoder of CLIP
        self.encoder = model.transformer

        # freeze all pretrained layers by setting requires_grad=False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # add a linear layer
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, x):
        # text encoder
        with torch.no_grad():
            x = self.encoder(x)
        # ---

        # linear head
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # ---

        return x


# %%
net = CLIP_SF_image_encoder().to(device)
summary(
    net,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
)


# %%
class CLIP_SF(torch.nn.Module):
    def __init__(
        self,
        visual_encoder: Callable[[Float[torch.Tensor, "X 3 244 244"]], Float[torch.Tensor, "X 1024"]],  # visual encoder
        visual_encoder_preprocess,  # visual encoder preprocessing
        text_encoder: Callable[[Float[torch.Tensor, "P 77"]], Float[torch.Tensor, "P 1024"]],  # natural language prompts encoder
        text_encoder_preprocess,  # text encoder preprocessing
        freeze_visual_encoder: bool = False,  # TRUE -> visual encoder parameters are not trained
        freeze_text_encoder: bool = False,  # TRUE -> text encoder parameters are not trained
    ):
        super().__init__()

        # todo: aggiungere controllo che non posssono essere entrambi frizzati gli encoder, altrimenti non ho gradient e ottengo un errore

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.visual_encoder_preprocess = visual_encoder_preprocess
        self.text_encoder_preprocess = text_encoder_preprocess
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_text_encoder = freeze_text_encoder

        """
    if freeze_visual_encoder:
      for param in self.visual_encoder.parameters():  # freeze all pretrained layers by setting requires_grad=False
        param.requires_grad = False

    if freeze_text_encoder:
      for param in self.text_encoder.parameters():    # freeze all pretrained layers by setting requires_grad=False
        param.requires_grad = False
    """

    # preprocess input prompts as required by the visual encoder
    def visual_preprocess(self, input_crops: list[PIL.Image]) -> Float[torch.Tensor, "X 3 244 244"]:
        output_crops: Float[torch.Tensor, "X 3 244 244"] = torch.stack(
            [self.visual_encoder_preprocess(crop) for crop in input_crops]
        ).to(device=device)

        return output_crops

    # preprocess text prompts as required by the text encoder
    def text_preprocess(self, input_prompts: list[str]) -> Int[torch.Tensor, "P 77"]:
        ##########print("input_prompts")
        ##########print(input_prompts)
        output_prompts: Int[torch.Tensor, "P 77"] = self.text_encoder_preprocess(
            input_prompts
        )

        return output_prompts

    def cosine_similarity(self, images_z: Float[torch.Tensor, "X 4"], texts_z: Float[torch.Tensor, "1 4"]) -> Float[torch.Tensor, "X"]:
        # normalise the image and the text
        images_z = images_z / images_z.norm(dim=-1, keepdim=True)
        texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

        # evaluate the cosine similarity between the sets of features
        similarity = texts_z @ images_z.T

        return similarity.cpu()

    def forward(self, x: list[tuple[list[PIL.Image], list[str]]]) -> list[Float[torch.Tensor, "dunno"]]:
        # x :: [([crop11, crop12, ..., crop1M], [prompt_11, ..., prompt_1N]), ([crop21, crop22, ..., crop2P], [prompt_21, ..., prompt_2K])]

        y: list[Float[torch.Tensor, "dunno"]] = list()

        input_x_crop_list: list[PIL.Image]
        input_x_prompt_list: list[str]

        for input_x_crop_list, input_x_prompt_list in x:
            # input_x :: ([crop11, crop12, ..., crop1M], [prompt_11, ..., prompt_1N])

            ##########print(len(input_x_crop_list))
            ##########print(input_x_prompt_list)

            # step 1: preprocess crops as required by the visual encoder
            with torch.no_grad():
                input_x_crop_list_preprocessed: Float[torch.Tensor, "X 3 244 244"] = self.visual_preprocess(input_x_crop_list)

            # step 2: preprocess prompts as required by the text encoder
            with torch.no_grad():
                input_x_prompt_list_preprocessed: Int[torch.Tensor, "P 77"] = self.text_preprocess(input_x_prompt_list)

            ##########print("input_x_crop_list_preprocessed.shape")
            ##########print(input_x_crop_list_preprocessed.shape)
            ##########print("input_x_prompt_list_preprocessed.shape")
            ##########print(input_x_prompt_list_preprocessed.shape)

            # step 3: compute crop representation in the latent space
            if self.freeze_visual_encoder:
                with torch.no_grad():
                    crop_list_z: Float[torch.Tensor, "X 1024"] = self.visual_encoder(input_x_crop_list_preprocessed)
            else:
                crop_list_z: Float[torch.Tensor, "X 1024"] = self.visual_encoder(input_x_crop_list_preprocessed)

            ##########print()
            ##########print("crop_list_z")
            ##########print(crop_list_z)
            ##########print(crop_list_z.shape)

            # step 4: compute prompt representation in the latent space
            if self.freeze_text_encoder:
                with torch.no_grad():
                    prompt_list_z: Float[torch.Tensor, "P 1024"] = self.text_encoder(input_x_prompt_list_preprocessed)
            else:
                prompt_list_z: Float[torch.Tensor, "P 1024"] = self.text_encoder(input_x_prompt_list_preprocessed)

            ##########print()
            ##########print("prompt_list_z")
            ##########print(prompt_list_z)
            ##########print(prompt_list_z.shape)

            # step 5: evaluate logits
            similarity_matrix: Float[torch.Tensor, "X P"] = self.cosine_similarity(
                prompt_list_z, crop_list_z
            )  # todo: valutare se usare torch.nn.functional.cosine_similarity

            ##########print("\nSIMILARITY MATRIX")
            ##########print(similarity_matrix)
            ##########print()

            # step 6: get prediction
            mean_similarity_bbox: Float[torch.Tensor, "X"] = torch.mean(similarity_matrix, dim=1)

            ##########print("mean_similarity_bbox")
            ##########print(mean_similarity_bbox)

            y.append(mean_similarity_bbox)

        # return torch.tensor(y, dtype=torch.float32)  # todo: togliere quando ne saremo sicuri
        return y


# %%
def get_optimizer(model, _lr, _wd, _momentum):
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=_lr, weight_decay=_wd, momentum=_momentum
    )

    return optimizer


# %%
def get_cost_function():
    def iou_loss(bbox_prediction, bbox_groundtruth):
        # compute intersection over union between ground truth bboxes and predicted bboxes
        iou_loss_matrix = torchvision.ops.box_iou(bbox_prediction, bbox_groundtruth)

        # extract the diagonal elements
        iou_loss_matrix_diagonal = torch.diag(iou_loss_matrix)

        # compute the mean of the intersection over union
        mean_iou = iou_loss_matrix_diagonal.mean()

        # compute the iou error
        iou_loss_output = 1 - mean_iou

        return iou_loss_output

    # INPUT:
    #   y_pred: [tensor([bbox1, bbox2, ..., bboxM]), tensor([bbox1, bbox2, ..., bboxN])] such that: len(y_pred) = batch_size and each tensor represent the similarities between bounding box i and the prompt
    #   y: tensor([1,2]) ground truth bbox. In this example the right bbox for first batch is the one in position 1 and in the second batch is the one in position 2
    # OUTPUT:
    #  avg(torch.nn.CrossEntropyLoss) such that the cross entropy loss is computed for each batch element separately in order to deal with different number of classes
    def cross_entropy(y_pred, y):
        avg_loss = 0.0

        for batch_item_pred, batch_item_ground_truth in zip(y_pred, y):
            ln = torch.nn.CrossEntropyLoss()(batch_item_pred, batch_item_ground_truth)
            avg_loss = avg_loss + ln

        avg_loss = avg_loss / len(y_pred)

        return avg_loss

    return cross_entropy


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
# deal with different size tensors in dataloader
def custom_collate(
    batch,
) -> tuple[list[Img], list[str], list[Float[torch.Tensor, "4"]]]:
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]

    return images, prompts, bboxes


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
# input:
#   -> retrived_bboxes : bounding boxes proposed by the region proposal model
#   -> bbox_groundtruth : ground truth bounding box provided by the training sample
# output:
#   -> [3, 5] in this case the for the first element in the batch the best bbox is the fourth, while for the second element in the batch the best bbox is the sixth. The best bbox is the one characterized by the largest IoU with the ground truth bbox
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
def training_step(
    model: torch.nn.Module,  # neural network to be trained
    region_proposal_model: torch.nn.Module,  # region proposal model
    data_loader: torch.utils.data.DataLoader,  # [train_dataset]
    loss_fn: torch.nn.Module,  # todo: in our case it is not correct nn.Module, test data type
    optimizer: torch.optim.Optimizer,  # optimizer
    accuracy_fn,  # accuracy function
    device: torch.device = device,  # target device
):
    train_loss = 0.0
    iou_train_acc = (
        0.0  # todo riflettere se aggiungere anche le altre accuracy? tipo semantic ?
    )

    model.to(device)
    model.train()

    for batch_idx, (img, prompts, true_xywh) in enumerate(tqdm(data_loader)):
        """
        print()
        print(f"batch_idx: {batch_idx}")
        print(f"img: {img}")
        print(f"type img: {type(img)}")
        print(f"prompts: {prompts}")
        print(f"true_xywh: {true_xywh}")
        """

        # send data to target device
        # convert bbox to the proper format
        true_xywh = torch.stack((true_xywh)).to(device)

        ##########print("\n pre true_xywh")
        ##########print(true_xywh)
        ##########print("")

        [true_xyxy] = torchvision.ops.box_convert(
            true_xywh.unsqueeze(0), in_fmt="xywh", out_fmt="xyxy"
        )

        ##########print("\n post true_xyxy")
        ##########print(true_xyxy)
        ##########print("")

        # forward pass

        with torch.no_grad():
            # i. region proposal
            bboxes = region_proposal_model(img)

            ##########print(f"\n\n REGION PROPOSAL ALGORITHM DONE bounding boxes: {bboxes}\n\n")

            # ii. get best bounding box with respect to the ground truth
            bbox_groundtruth = best_bbox_one_hot_encoding(bboxes, true_xyxy)

            # from yolo bboxes to cropped images
            crops = []
            for batch_image_pil, batch_image_bboxes in zip(img, bboxes):

                list_bboxes_image: list[Image] = [
                    batch_image_pil.crop((xmin, ymin, xmax, ymax))
                    for bbox in batch_image_bboxes
                    for [xmin, ymin, xmax, ymax, _, _] in [bbox.tolist()]
                ]

                crops.append(list_bboxes_image)

            # prepare neural network input
            model_input = list(zip(crops, prompts))

        # forward pass
        model_output = net(model_input)

        ##########print("\n\nMODEL OUTPUT FINALLY (-:")
        ##########print("model_output")
        ##########print(model_output)

        # calculate loss
        loss = loss_fn(model_output, bbox_groundtruth)
        train_loss += loss

        ##########print("loss train")
        ##########print(loss)

        # get index of the predicted bounding box in order to compute IoU accuracy
        bbox_index_pred = torch.tensor(
            [torch.argmax(batch_item_pred) for batch_item_pred in model_output]
        )

        # get predicted bounding box for each example in the batch
        bbox_pred = [
            batch_example_bboxes[idx]
            for batch_example_bboxes, idx in zip(bboxes, bbox_index_pred)
        ]
        bbox_pred = torch.stack(bbox_pred)

        # debug: display prediction of first element in the batch
        ##########print("true_xywh[0]")
        ##########print(true_xywh[0])
        ##########print("bbox_pred[0][:4]")
        ##########print(bbox_pred[0][:4])
        prediction_obj = Prediction(img[0], prompts[0], true_xyxy[0], bbox_pred[0][:4])
        display_predictions([prediction_obj])

        # calculate intersection over union train accuracy
        acc = accuracy_fn(bbox_pred, true_xyxy)
        iou_train_acc += acc

        ##########print("acc train")
        ##########print(acc)

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

    ##########print("num_iteration = "+str(num_iteration))
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= (batch_idx + 1)
    iou_train_acc /= (batch_idx + 1)
    print(f"Train loss: {train_loss:.5f} | IoU train accuracy: {iou_train_acc}")
    return train_loss, iou_train_acc


# %%
def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    region_proposal_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, iou_test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode

    # Turn on inference context manager
    with torch.inference_mode():
        for batch_idx, (img, prompts, true_xywh) in enumerate(tqdm(data_loader)):

            # get ground truth bbox tensor
            true_xywh = torch.stack((true_xywh)).to(device)

            # convert bbox to the proper format
            [true_xyxy] = torchvision.ops.box_convert(
                true_xywh.unsqueeze(0), in_fmt="xywh", out_fmt="xyxy"
            )

            # forward pass
            # i. region proposal
            bboxes = region_proposal_model(img)

            # ii. get best bounding box with respect to the ground truth
            bbox_groundtruth = best_bbox_one_hot_encoding(bboxes, true_xyxy)

            # from yolo bboxes to cropped images
            crops = []
            for batch_image_pil, batch_image_bboxes in zip(img, bboxes):

                list_bboxes_image: list[Image] = [
                    batch_image_pil.crop((xmin, ymin, xmax, ymax))
                    for bbox in batch_image_bboxes
                    for [xmin, ymin, xmax, ymax, _, _] in [bbox.tolist()]
                ]

                crops.append(list_bboxes_image)

            # prepare neural network input
            model_input = list(zip(crops, prompts))

            # forward pass
            model_output = net(model_input)

            # calculate loss
            loss = loss_fn(model_output, bbox_groundtruth)
            test_loss += loss

            ##########print("loss test")
            ##########print(loss)

            # get index of the predicted bounding box in order to compute IoU accuracy
            bbox_index_pred = torch.tensor(
                [torch.argmax(batch_item_pred) for batch_item_pred in model_output]
            )

            # get predicted bounding box for each example in the batch
            bbox_pred = [
                batch_example_bboxes[idx]
                for batch_example_bboxes, idx in zip(bboxes, bbox_index_pred)
            ]
            bbox_pred = torch.stack(bbox_pred)

            # calculate intersection over union train accuracy
            acc = accuracy_fn(bbox_pred, true_xyxy)
            iou_test_acc += acc

            ##########print("acc test")
            ##########print(acc)

        # Adjust metrics and print out
        test_loss /= (batch_idx + 1)
        iou_test_acc /= (batch_idx + 1)
        print(f"Test loss: {test_loss:.5f} | IoU test accuracy: {iou_test_acc:.5f}\n")
        return test_loss, iou_test_acc


# %% [markdown]
# Put all together in a training loop.

# %%
# tensorboard logging utilities
def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


# %%
# load clip model and retrieve textual and visual encoders
clip_model, preprocess = clip.load("RN50")
clip_model = clip_model.to(device=device).eval()
# clip_visual_encoder = clip_model.visual
# clip_visual_encoder = clip_visual_encoder.to(device=device).eval()
clip_visual_encoder = clip_model.encode_image
# clip_text_encoder = clip_model.transformer
# clip_text_encoder = clip_text_encoder.to(device=device).eval()
clip_text_encoder = clip_model.encode_text

# %%
# personalized visual encoder
sf_visual_encoder = CLIP_SF_image_encoder().to(device)

# %%
# personalized text encoder
sf_text_encoder = CLIP_SF_text_encoder().to(device)

# %%
# flag configuration to decide the encoders to be trained
freeze_visual_encoder = False
freeze_text_encoder = True

# instantiate the network and move it to the chosen device (GPU)
net = CLIP_SF(
    visual_encoder=sf_visual_encoder,
    visual_encoder_preprocess=preprocess,
    text_encoder=clip_text_encoder,
    text_encoder_preprocess=clip.tokenize,
    freeze_visual_encoder=freeze_visual_encoder,
    freeze_text_encoder=freeze_text_encoder,
).to(device)

# %%
# instantiate the region proposal algorithm
yolo = Yolo_v5().to(device)

# %%
# setting a manual seed allow us to provide reprudicible results in this notebook
torch.manual_seed(42)

# measure time
train_time_start = (
    timer()
)  # todo: forse misurando il tempo possiamo far apprezzare la differenza di tempo di esecuzione del training quando abbiamo fatto il preprocessing delle bounding box vs senza

# create a logger for the experiment
writer = SummaryWriter(log_dir="runs/exp1")

# get dataloaders
BATCH_SIZE = 2
NUM_WORKERS = os.cpu_count()  # TODO: non va con questo
# NUM_WORKERS = 1

# get dataset instance
train_dataset: Dataset[tuple[Img, list[str], UInt[torch.Tensor, "4"]]] = CocoDataset(
    split="train", limit=BATCH_SIZE * 5
)
test_dataset: Dataset[tuple[Img, list[str], UInt[torch.Tensor, "4"]]] = CocoDataset(
    split="test", limit=BATCH_SIZE * 5
)
val_dataset: Dataset[tuple[Img, list[str], UInt[torch.Tensor, "4"]]] = CocoDataset(
    split="val", limit=BATCH_SIZE * 5
)
print(
    f"LEN_TRAIN_DATASET: {len(train_dataset)}, LEN_TEST_DATASET: {len(test_dataset)}, LEN_VALIDATION_DATASET: {len(val_dataset)}"
)

print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate,
    shuffle=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate,
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
cost_function = get_cost_function()

# define the accuracy function
accuracy_fn = get_accuracy_function()

# computes evaluation results before training
"""
print('Before training:')
train_loss, train_accuracy = test_step(model = net,
        region_proposal_model = yolo,
        data_loader = train_loader,
        loss_fn = cost_function,
        accuracy_fn = accuracy_fn,
        max_sample = 5)
val_loss, val_accuracy = test_step(model = net,
        region_proposal_model = yolo,
        data_loader = val_loader,
        loss_fn = cost_function,
        accuracy_fn = accuracy_fn,
        max_sample = 5)
test_loss, test_accuracy = test_step(model = net,
        region_proposal_model = yolo,
        data_loader = test_loader,
        loss_fn = cost_function,
        accuracy_fn = accuracy_fn,
        max_sample = 5)

# log to TensorBoard
log_values(writer, -1, train_loss, train_accuracy, "train")
log_values(writer, -1, val_loss, val_accuracy, "validation")
log_values(writer, -1, test_loss, test_accuracy, "test")

print('\tTraining loss {:.5f}, Training accuracy {:.5f}'.format(train_loss, train_accuracy))
print('\tValidation loss {:.5f}, Validation accuracy {:.5f}'.format(val_loss, val_accuracy))
print('\tTest loss {:.5f}, Test accuracy {:.5f}'.format(test_loss, test_accuracy))
print('-----------------------------------------------------')
"""
epochs = 3
for epoch in tqdm(range(epochs)):
    train_loss, train_accuracy = training_step(
        model=net,
        region_proposal_model=yolo,
        data_loader=train_loader,
        loss_fn=cost_function,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
    )

    val_loss, val_accuracy = test_step(
        model=net,
        region_proposal_model=yolo,
        data_loader=val_loader,
        loss_fn=cost_function,
        accuracy_fn=accuracy_fn,
    )

    # logs to TensorBoard
    log_values(writer, epoch, train_loss, train_accuracy, "train")
    log_values(writer, epoch, val_loss, val_accuracy, "validation")

    print("Epoch: {:d}".format(epoch + 1))
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
    print("-----------------------------------------------------")

train_time_end = timer()
total_train_time_model_1 = print_train_time(
    start=train_time_start, end=train_time_end, device=device
)
# compute final evaluation results
print("After training:")
train_loss, train_accuracy = test_step(
    model=net,
    region_proposal_model=yolo,
    data_loader=train_loader,
    loss_fn=cost_function,
    accuracy_fn=accuracy_fn,
)
val_loss, val_accuracy = test_step(
    model=net,
    region_proposal_model=yolo,
    data_loader=val_loader,
    loss_fn=cost_function,
    accuracy_fn=accuracy_fn,
)
test_loss, test_accuracy = test_step(
    model=net,
    region_proposal_model=yolo,
    data_loader=test_loader,
    loss_fn=cost_function,
    accuracy_fn=accuracy_fn,
)

# log to TensorBoard
log_values(writer, epochs, train_loss, train_accuracy, "train")
log_values(writer, epochs, val_loss, val_accuracy, "validation")
log_values(writer, epochs, test_loss, test_accuracy, "test")

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
