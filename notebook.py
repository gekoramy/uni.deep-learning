# %%
# %%shell
if ! [ -d dataset ]; then
  mkdir dataset &&
  gdown 1i-LHWSRp2F6--yhAi4IG3DiiCHmgE4cw &&
  tar -xf refcocog.tar -C dataset &&
  rm refcocog.tar
fi

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
tensorboard
END

pip install -q -r requirements.txt
pip install -q git+https://github.com/openai/CLIP.git

# %%
# %load_ext tensorboard

# %%
# %tensorboard --logdir ./runs

# %% [markdown]
# # Standard finetuning
# In this notebook we propose the straightforward solution to fine tune CLIP. The general idea is to add linear layer(s) on top of the 1024 visual features of CLIP.

# %% [markdown]
# Dependences

# %%
import doctest
import clip
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import itertools as it
import typing as t
import csv

from collections import defaultdict
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, ConvertImageDtype
from torchvision.transforms.functional import crop
from torchinfo import summary
from tqdm.notebook import tqdm, trange
from torchvision.ops import box_iou, box_convert

# %%
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# %%
# setting a manual seed allow us to provide reprudicible results in this notebook
# https://pytorch.org/docs/stable/notes/randomness.html

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(False)  # CLIP uses non-deterministic algorithms
g: torch.Generator = torch.Generator(device=device).manual_seed(42)

# %% [markdown]
# ## Load the dataset
# First of all we have to load the dataset.

# %% [markdown]
# Folder paths

# %%
path_root: str = os.path.join("dataset", "refcocog", "")
path_annotations: str = os.path.join(path_root, "annotations", "")
path_bboxes: str = os.path.join(path_root, "bboxes", "")
path_images: str = os.path.join(path_root, "images", "")

path_refs: str = os.path.join(path_annotations, "refs.csv")
path_sentences: str = os.path.join(path_annotations, "sentences.csv")

path_DETR: str = os.path.join(path_bboxes, "bboxes[DETR].csv")
path_YOLOv5: str = os.path.join(path_bboxes, "bboxes[YOLOv5].csv")
path_YOLOv8: str = os.path.join(path_bboxes, "bboxes[YOLOv8].csv")

# %% [markdown]
# Type declaration

# %%
Split = t.Literal["train", "test", "val"]


@dataclass
class Ref:
    ref_id: int  # unique id for refering expression
    file_name: str  # file name of image relative to img_root
    split: Split
    xmin: float
    ymin: float
    xmax: float
    ymax: float


with open(path_refs, "r") as f:
    raw = csv.DictReader(f)
    refs: list[Ref] = [Ref(**row) for row in raw]

# %%
T = t.TypeVar("T")
K = t.TypeVar("K")
V = t.TypeVar("V")


def groupby(
    xs: list[T],
    map_key: t.Callable[[T], K],
    map_value: t.Callable[[T], V] = lambda x: x,
) -> dict[K, list[V]]:
    return {
        k: [map_value(v) for v in vs]
        for k, vs in it.groupby(sorted(xs, key=map_key), key=map_key)
    }


# %%
@dataclass
class Sentence:
    ref_id: int  # unique id for refering expression
    sent: str


with open(path_sentences, "r") as f:
    raw = csv.DictReader(f)
    sentences: list[Sentence] = [Sentence(**row) for row in raw]


id2sents: dict[int, list[str]] = groupby(
    sentences, lambda x: x.ref_id, lambda x: x.sent
)


# %%
@dataclass
class BBox:
    file_name: str  # file name of image relative to img_root
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float


with open(path_DETR, "r") as f:
    raw = csv.DictReader(f)
    bboxes: list[BBox] = [BBox(**row) for row in raw]

img2detr: dict[str, list[BBox]] = defaultdict(
    list, groupby(bboxes, lambda x: x.file_name)
)


with open(path_YOLOv5, "r") as f:
    raw = csv.DictReader(f)
    bboxes: list[BBox] = [BBox(**row) for row in raw]

img2yolov5: dict[str, list[BBox]] = defaultdict(
    list, groupby(bboxes, lambda x: x.file_name)
)


with open(path_YOLOv8, "r") as f:
    raw = csv.DictReader(f)
    bboxes: list[BBox] = [BBox(**row) for row in raw]

img2yolov8: dict[str, list[BBox]] = defaultdict(
    list, groupby(bboxes, lambda x: x.file_name)
)

# %% [markdown]
# Define custom dataset

# %%
TensorImage = UInt[torch.Tensor, "3 W H"]

class CocoDataset(
    Dataset[
        tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]
    ]
):
    def __init__(
        self,
        split: Split,
        img2bboxes: dict[str, list[BBox]],
        limit: int = -1,
    ):
        self.items: list[
            tuple[
                str, list[str], Float[torch.Tensor, "X 5"], Float[torch.Tensor, "1 4"]
            ]
        ] = [
            (img, sents, xyxys, xyxy)
            for ref in refs
            if ref.split == split
            for img in [os.path.join(path_images, ref.file_name)]
            for sents in [id2sents[ref.ref_id]]
            for bboxes in [img2bboxes[ref.file_name]]
            for xyxys in [
                torch.tensor([
                    (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
                    for bbox in bboxes
                    if bbox.confidence > .25  # ensure at least .25 confidence
                ])
            ]
            for xyxy in [
                torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)])
            ]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))


    def __len__(self) -> int:
        return self.len


    def __getitem__(
        self, index: int
    ) -> tuple[
        TensorImage, list[str], Float[torch.Tensor, "X 5"], Float[torch.Tensor, "1 4"]
    ]:
        file_name, sents, xyxys, xyxy = self.items[index]
        return read_image(file_name, ImageReadMode.RGB).to(device), sents, xyxys, xyxy


# %%
class CocoTrainDataset(Dataset[tuple[list[TensorImage], list[str], int]]):

    def __init__(
        self,
        split: Split,
        img2bboxes: dict[str, list[BBox]],
        limit: int = -1,
    ):
        self.items: list[
            tuple[str, list[str], Float[torch.Tensor, "X 4"], int]
        ] = [
            (img, sents, xyxys, i)
            for ref in refs
            if ref.split == split
            for img in [os.path.join(path_images, ref.file_name)]
            for sents in [id2sents[ref.ref_id]]
            for bboxes in [img2bboxes[ref.file_name]]
            if len(bboxes) > 1  # ensure at least 2 bboxes per image
            for xyxys in [
                torch.tensor([
                    (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
                    for bbox in bboxes
                    if bbox.confidence > .25  # ensure at least .25 confidence
                ])
            ]
            for xyxy in [
                torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)])
            ]
            for ious in [box_iou(xyxys, xyxy)]
            if torch.max(ious).item() > .5  # ensure at least .5 of IoU
            for i in [torch.argmax(ious).item()]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))


    def __len__(self) -> int:
        return self.len


    def __getitem__(
        self, index: int
    ) -> tuple[list[TensorImage], list[str], int]:
        file_name, sents, xyxys, i = self.items[index]
        img: TensorImage = read_image(file_name, ImageReadMode.RGB).to(device)

        xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt= 'xyxy', out_fmt= 'xywh').int()  # TODO cosa farebbe PIL (bboxes con float)?

        crops: list[TensorImage] = [
            crop(img, top=x, left=y, height=h, width=w)
            for xywh in xywhs
            for [x, y, w, h] in [xywh.tolist()]
        ]

        return crops, sents, i



# %% [markdown]
# ## Region proposal networks

# %% [markdown]
# ## Linear layers on top of image encoder

# %% [markdown]
# In the following cell we create a neural network that builds a linear head on top of the visual encoder of CLIP.

# %%
class CLIP_freezed_img_encoder(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()
        clip_model, _ = clip.load("RN50", device=device)
        self.encode_image = clip_model.encode_image

        for p in clip_model.parameters():
            p.requires_grad = False


    def forward(self, image: Float[torch.Tensor, "crops 3 244 244"]) -> Float[torch.Tensor, "crops 1024"]:
        with torch.no_grad():
            return self.encode_image(image).float()


class CLIP_freezed_txt_encoder(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()
        clip_model, _ = clip.load("RN50", device=device)
        self.encode_text = clip_model.encode_text

        for p in clip_model.parameters():
            p.requires_grad = False


    def forward(self, text: Int[torch.Tensor, "prompts 77"]) -> Float[torch.Tensor, "prompts 1024"]:
        with torch.no_grad():
            return self.encode_text(text).float()



# %%
clip_freezed_img_encoder: CLIP_freezed_img_encoder = CLIP_freezed_img_encoder(device=device)
clip_freezed_txt_encoder: CLIP_freezed_txt_encoder = CLIP_freezed_txt_encoder(device=device)


# %%
class CLIP_SF_img_encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
           clip_freezed_img_encoder,
           nn.Linear(1024, 1024),
           nn.ReLU(),
           nn.Linear(1024, 1024),
        )


class CLIP_SF_txt_encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
           clip_freezed_txt_encoder,
           nn.Linear(1024, 1024),
           nn.ReLU(),
           nn.Linear(1024, 1024),
        )



# %%
summary(
    CLIP_SF_img_encoder().to(device),
    input_size=(1, 3, 224, 224),
    dtypes=[torch.float],
    device=device,
    col_names=["input_size", "output_size", "num_params", "trainable"],
)

# %%
summary(
    CLIP_SF_txt_encoder().to(device),
    input_size=(1, 77),
    dtypes=[torch.int],
    col_names=["input_size", "output_size", "num_params", "trainable"],
    device=device,
)


# %%
class CLIP_SF_CORE(nn.Module):

    def __init__(
        self,
        visual_encoder: nn.Module,  # visual encoder
        text_encoder: nn.Module,  # natural language prompts encoder
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder


    def cosine_similarity(self, crops_z: Float[torch.Tensor, "crops 1024"], prompts_z: Float[torch.Tensor, "prompts 1024"]) -> Float[torch.Tensor, "crops prompts"]:
        # normalise the image and the text
        crops_z = crops_z / crops_z.norm(dim=-1, keepdim=True)
        prompts_z = prompts_z / prompts_z.norm(dim=-1, keepdim=True)

        # evaluate the cosine similarity between the sets of features
        return prompts_z @ crops_z.T


    def forward(self, crops: Float[torch.Tensor, "crops 3 244 244"], prompts: Int[torch.Tensor, "prompts 77"]) -> Float[torch.Tensor, "crops 1"]:

        # step 1: compute crop representation in the latent space
        crop_z: Float[torch.Tensor, "crops 1024"] = self.visual_encoder(crops)

        # step 2: compute prompt representation in the latent space
        prompt_z: Int[torch.Tensor, "prompts 1024"] = self.text_encoder(prompts)

        # step 3: evaluate logits
        similarity_matrix: Float[torch.Tensor, "crops prompts"] = self.cosine_similarity(crop_z, prompt_z)

        # step 4: crops classification
        return torch.mean(similarity_matrix, dim=0)



# %%
class CLIP_SF(nn.Module):

    def __init__(
        self,
        img_preprocess: t.Callable[[TensorImage], Float[torch.Tensor, "3 244 244"]],
        txt_preprocess: t.Callable[[list[str]], Float[torch.Tensor, "prompts 77"]],
        img_encoder: nn.Module,  # visual encoder
        txt_encoder: nn.Module,  # natural language prompts encoder
    ):
        super().__init__()
        self.img_preprocess = img_preprocess
        self.txt_preprocess = txt_preprocess
        self.core = CLIP_SF_CORE(img_encoder, txt_encoder)


    def forward(self, crops: list[TensorImage], prompts: list[str]) -> Float[torch.Tensor, "crops 1"]:

        # step 1: preprocess crops as required by the visual encoder
        with torch.no_grad():
            crops_preprocessed: Float[torch.Tensor, "crops 3 244 244"] = torch.stack([
                self.img_preprocess(crop)
                for crop in crops
            ])

        # step 2: preprocess prompts as required by the text encoder
        with torch.no_grad():
            prompts_preprocessed: Int[torch.Tensor, "prompts 77"] = self.txt_preprocess(prompts)

        return self.core(crops_preprocessed, prompts_preprocessed)



# %%
with SummaryWriter() as writer:

    writer.add_graph(
        model=CLIP_SF_CORE(
            visual_encoder=CLIP_SF_img_encoder(),
            text_encoder=CLIP_SF_txt_encoder(),
        ),
        input_to_model=[
            torch.ones((5, 3, 244, 244)),
            torch.ones((2, 77), dtype=torch.int),
        ]
    )



# %%
summary(
    CLIP_SF_CORE(
        visual_encoder=CLIP_SF_img_encoder(),
        text_encoder=CLIP_SF_txt_encoder(),
    ),
    input_size=[(5, 3, 244, 244), (2, 77)],
    dtypes=[torch.float, torch.int],
    col_names=["input_size", "output_size", "num_params", "trainable"],
)


# %%
def unzip(batch: list[tuple[T, ...]]) -> tuple[tuple[T, ...], ...]:
    """

    >>> unzip([('A', 1), ('B', 2)])
    (('A', 'B'), (1, 2))

    """
    return tuple(zip(*batch))



# %%
def bext_bbox(pred: Float[torch.Tensor, "crops 4"], groundtruth: Float[torch.Tensor, "1 4"]) -> int:
    """

    >>> bext_bbox(
    ...     torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 2, 2]]),
    ...     torch.tensor([[0, 0, 1, 1]])
    ... )
    0

    >>> bext_bbox(
    ...     torch.tensor([[0, 0, 0, 0], [0, 0, 2, 2], [1, 1, 2, 2]]),
    ...     torch.tensor([[0, 0, 1, 1]])
    ... )
    1

    """
    return torch.argmax(box_iou(pred, groundtruth)).item()



# %%
doctest.testmod()

# %%
loss_fn: t.Callable[[Float[torch.Tensor, 'crops'], Int[torch.Tensor, '1']], Float[torch.Tensor, '1']] = \
    nn.functional.cross_entropy


# %%
def training_step(
    model: nn.Module,
    data_loader: DataLoader[tuple[list[TensorImage], list[str], int]],
    optimizer: torch.optim.Optimizer,
) -> float :

    model.train()

    running_loss: float = 0
    progress = tqdm(data_loader, desc=f"training")

    cropss: tuple[tuple[TensorImage, ...], ...]
    promptss: tuple[list[str], ...]
    true_is: tuple[int, ...]

    for i, (cropss, promptss, true_is) in enumerate(progress):

        # forward pass
        preds: list[Float[torch.Tensor, "crops"]] = [
            model(crops, prompts)
            for crops, prompts in zip(cropss, promptss)
        ]

        # calculate loss
        losses: Float[torch.Tensor, "batch"] = torch.stack([
            loss_fn(pred, torch.tensor(true_i))
            for pred, true_i in zip(preds, true_is)
        ])
        loss: Float[torch.Tensor, '1'] = torch.mean(losses)
        running_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        progress.set_postfix({ "loss": running_loss / (i + 1) }, refresh=False)

    return running_loss / len(data_loader)



# %%
def test_step(
    model: nn.Module,
    data_loader: DataLoader[tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]],
) -> tuple[float, float] :

    model.eval()

    running_loss: float = 0
    running_acc: float = 0
    progress = tqdm(data_loader, desc=f"testing")

    with torch.inference_mode():

        img: TensorImage
        prompts: list[str]
        xyxys: Float[torch.Tensor, "crops 4"]
        xyxy: Float[torch.Tensor, "4"]

        for iter, (img, prompts, xyxys, true_xyxy) in zip(it.count(1), progress):

            true_i: int = bext_bbox(xyxys, true_xyxy)

            # from xyxys to crops
            xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt= 'xyxy', out_fmt= 'xywh').int()  # TODO: cosa farebbe PIL?

            crops: list[TensorImage] = [
                crop(img, top=x, left=y, height=h, width=w)
                for xywh in xywhs
                for [x, y, w, h] in [xywh.tolist()]
            ]

            # forward pass
            model_output: Float[torch.Tensor, 'crops'] = model(crops, prompts)

            # calculate loss
            loss: float = loss_fn(model_output, torch.tensor(true_i)).item()
            running_loss += loss

            # calculate IoU accuracy

            # # get index of the predicted bounding box to compute IoU accuracy
            pred_i: int = torch.argmax(model_output).item()

            # # get predicted bounding
            pred_xyxy: Float[torch.Tensor, "4"] = xyxys[pred_i]

            # # IoU
            acc: float = box_iou(true_xyxy, pred_xyxy.unsqueeze(0)).item()
            running_acc += acc

            progress.set_postfix(
                {
                    "loss": running_loss / iter,
                    "iou": running_acc / iter,
                },
                refresh=False
            )

        return running_loss / len(data_loader), running_acc / len(data_loader)



# %%
def showtime(
    model: nn.Module,
    data_loader: DataLoader[tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]],
    writer: SummaryWriter,
    global_step: int,
) -> None:

    model.eval()

    with torch.inference_mode():

        img: TensorImage
        prompts: list[str]
        xyxys: Float[torch.Tensor, "crops 4"]
        xyxy: Float[torch.Tensor, "4"]

        progress = tqdm(data_loader, desc=f"showtime")

        for iter, (img, prompts, xyxys, true_xyxy) in enumerate(progress):

            true_i: int = bext_bbox(xyxys, true_xyxy)

            # from xyxys to crops
            xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt= 'xyxy', out_fmt= 'xywh').int()

            crops: list[TensorImage] = [
                crop(img, top=x, left=y, height=h, width=w)
                for xywh in xywhs
                for [x, y, w, h] in [xywh.tolist()]
            ]

            # forward pass
            model_output: Float[torch.Tensor, 'crops'] = model(crops, prompts)

            # get index of the predicted bounding box to compute IoU accuracy
            pred_i: int = torch.argmax(model_output).item()

            # get predicted bounding
            pred_xyxy: Float[torch.Tensor, "4"] = xyxys[pred_i]

            # https://github.com/pytorch/pytorch/issues/65449
            writer.add_image_with_boxes(
                tag=f"{iter + 1}",
                img_tensor=img,
                box_tensor=torch.stack((xyxys[pred_i], xyxys[true_i], true_xyxy.squeeze())),
                labels=['prediction', 'best region proposal', 'ground truth'],
                global_step=global_step,
            )



# %% [markdown]
# Put all together in a training loop.

# %%
BATCH_SIZE: int = 512
LIMIT: int = 10 * BATCH_SIZE
NUM_WORKERS: int = 0  # os.cpu_count() or 1

train_loader: DataLoader[
    tuple[list[TensorImage], list[str], int]
] = DataLoader(
    dataset=CocoTrainDataset(split="train", img2bboxes=img2detr, limit=LIMIT),
    generator=g,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=unzip,
    shuffle=True,
)

val_loader: DataLoader[
    tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]
] = DataLoader(
    dataset=CocoDataset(split="val", img2bboxes=img2detr, limit=LIMIT),
    batch_size=None,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

test_loader: DataLoader[
    tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]
] = DataLoader(
    dataset=CocoDataset(split="test", img2bboxes=img2detr, limit=LIMIT),
    batch_size=None,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

showtime_loader: DataLoader[
    tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]
] = DataLoader(
    dataset=CocoDataset(split="test", img2bboxes=img2detr, limit=5),
    batch_size=None,
    num_workers=NUM_WORKERS,
    shuffle=False,
)


# %%
def training_loop(
    name: str,
    model: nn.Module,
    epochs: int,
    optimizer: t.Callable[[t.Iterable[torch.Tensor]], torch.optim.Optimizer]
) -> None:

    # create a logger for the experiment
    with SummaryWriter(f'runs/{name}') as writer:

        # computes evaluation results before training
        print('Before training:')
        test_loss, test_accuracy = test_step(
            model = model,
            data_loader = test_loader,
        )
        val_loss, val_accuracy = test_step(
            model=model,
            data_loader=val_loader,
        )

        showtime(
            model = model,
            data_loader = showtime_loader,
            writer = writer,
            global_step=0
        )

        # log to TensorBoard
        writer.add_scalars(
            main_tag='loss',
            tag_scalar_dict={
                'test': test_loss,
                'val': val_loss,
            },
            global_step=0,
        )
        writer.add_scalars(
            main_tag='accuracy',
            tag_scalar_dict={
                'test': test_accuracy,
                'val': val_accuracy,
            },
            global_step=0,
        )

        progress = trange(epochs, desc="epochs")
        for epoch in progress:

            train_loss = training_step(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer(model.parameters()),
            )

            val_loss, val_accuracy = test_step(
                model=model,
                data_loader=val_loader,
            )


            # log to TensorBoard
            writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={
                    'train': train_loss,
                    'val': val_loss,
                },
                global_step=epoch + 1,
            )
            writer.add_scalars(
                main_tag='accuracy',
                tag_scalar_dict={
                    'val': val_accuracy,
                },
                global_step=epoch + 1,
            )

            progress.set_postfix(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                },
                refresh=False
            )

        # compute final evaluation results
        print("After training:")

        test_loss, test_accuracy = test_step(
            model=model,
            data_loader=test_loader,
        )

        showtime(
            model = model,
            data_loader = showtime_loader,
            writer = writer,
            global_step=epochs,
        )

        # log to TensorBoard
        writer.add_scalars(
            main_tag='loss',
            tag_scalar_dict={
                'test': test_loss,
            },
            global_step=epochs,
        )
        writer.add_scalars(
            main_tag='accuracy',
            tag_scalar_dict={
                'test': test_accuracy,
            },
            global_step=epochs,
        )



# %%
def transform(n_px: int):
    """
    https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L75-L86
    """
    return Compose([
        ConvertImageDtype(torch.float),
        Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

preprocess: Compose = transform(244)

# %% [markdown]
# ## Architettura 1
# freeze immagine: false<br>
# freeze testo: true<br>
# layer interni immagine: 1024 -> 1024<br>
# layer interni testo: 1024 -> 1024<br>
# activation function immagine: ReLU<br>
# activation function testo: ReLU<br>
# optimizer: SGD(lr=0.01, weight_decay=0.000001, momentum=0.9)<br>
# epoch: 10<br>
# batch size: 16<br>

# %%
net1: CLIP_SF = CLIP_SF(
    img_preprocess=preprocess,
    txt_preprocess=clip.tokenize,
    img_encoder=nn.Sequential(
        clip_freezed_img_encoder,
        nn.ReLU(),
        nn.Linear(1024, 1024)
    ),
    txt_encoder=clip_freezed_txt_encoder,
)

# %%
training_loop(
    name="net1",
    model=net1,
    optimizer=lambda params: torch.optim.SGD(params=params, lr=1e-2, weight_decay=1e-6, momentum=.9),
    epochs=5,
)

# %% [markdown]
# ## Architettura 2
# freeze immagine: false<br>
# freeze testo: false<br>
# layer interni immagine: 1024 -> 512 -> 256 -> 1024<br>
# layer interni testo: 1024 -> 512 -> 256 -> 1024<br>
# activation function immagine: ReLU<br>
# activation function testo: ReLU<br>
# optimizer: SGD(lr=0.01, weight_decay=0.000001, momentum=0.9)<br>
# epoch: 10<br>
# batch size: 16<br>

# %%
net2: CLIP_SF = CLIP_SF(
    img_preprocess=preprocess,
    txt_preprocess=clip.tokenize,
    img_encoder=nn.Sequential(
        clip_freezed_img_encoder,
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1024),
    ),
    txt_encoder=nn.Sequential(
        clip_freezed_txt_encoder,
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1024),
    ),
)

# %%
training_loop(
    name="net2",
    model=net2,
    optimizer=lambda params: torch.optim.SGD(params=params, lr=1e-2, weight_decay=1e-6, momentum=.9),
    epochs=5,
)

# %% [markdown]
# ## Architettura 3
# freeze immagine: false<br>
# freeze testo: false<br>
# layer interni immagine: 1024 -> 512 -> 256 -> 1024<br>
# layer interni testo: 1024 -> 512 -> 256 -> 1024<br>
# activation function immagine: ReLU<br>
# activation function testo: Sigmoid<br>
# optimizer: Adagrad(come Nielsen lr=0.0015, weigth_decay=0.000001) eventualmente usare frase di nielsen e gif a questo sito come motivazione https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c<br>
# epoch: 10<br>
# batch size: 16<br>

# %%
net3: CLIP_SF = CLIP_SF(
    img_preprocess=preprocess,
    txt_preprocess=clip.tokenize,
    img_encoder=nn.Sequential(
        clip_freezed_img_encoder,
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1024),
    ),
    txt_encoder=nn.Sequential(
        clip_freezed_txt_encoder,
        nn.Sigmoid(),
        nn.Linear(1024, 512),
        nn.Sigmoid(),
        nn.Linear(512, 256),
        nn.Sigmoid(),
        nn.Linear(256, 1024),
    ),
)

# %%
training_loop(
    name="net3",
    model=net3,
    optimizer=lambda params: torch.optim.Adadelta(params=params, lr=15e-4, weight_decay=1e-6),
    epochs=5,
)

# %% [markdown]
# ## Architettura 4
# freeze immagine: false<br>
# freeze testo: false<br>
# layer interni immagine: 1024 -> 512<br>
# layer interni testo: 1024 -> 512<br>
# activation function immagine: ReLU<br>
# activation function testo: ReLU<br>
# optimizer: SGD(lr=0.01, weight_decay=0.000001, momentum=0.9)<br>
# epoch: 10<br>
# batch size: 16<br>

# %%
net4: CLIP_SF = CLIP_SF(
    img_preprocess=preprocess,
    txt_preprocess=clip.tokenize,
    img_encoder=nn.Sequential(
        clip_freezed_img_encoder,
        nn.ReLU(),
        nn.Linear(1024, 512),
    ),
    txt_encoder=nn.Sequential(
        clip_freezed_txt_encoder,
        nn.ReLU(),
        nn.Linear(1024, 512),
    ),
)

# %%
training_loop(
    name="net4",
    model=net4,
    optimizer=lambda params: torch.optim.SGD(params=params, lr=1e-2, weight_decay=1e-6, momentum=.9),
    epochs=5,
)
