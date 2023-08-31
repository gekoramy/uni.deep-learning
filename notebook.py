# %%
import csv
import itertools as it
import math
import os
import random
import typing as t

import clip
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from collections import defaultdict
from clip.model import CLIP
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.io import read_image, ImageReadMode
from torchvision.ops import box_iou, box_convert
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    Normalize,
    InterpolationMode,
    ConvertImageDtype,
    ColorJitter,
    GaussianBlur,
    RandomChoice,
    RandomPosterize,
    RandomSolarize,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomEqualize,
    Grayscale,
)
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm, trange

# %%
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# %%
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(False)  # CLIP uses non-deterministic algorithms
g: torch.Generator = torch.Generator(device=device).manual_seed(42)
random.seed(42)

# %% [markdown]
# ### CLIP

# %%
clip_model, clip_preprocessor = clip.load("RN50", device=device)
clip_model.float()
clip_model.eval()

for p in clip_model.parameters():
    p.requires_grad = False


# %%
def transform(n_px: int) -> Compose:
    """
    https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L75-L86
    """
    return Compose([
        ConvertImageDtype(torch.float),
        Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


preprocess: Compose = transform(224)


# %%
class ClipFrozenImgEnc(nn.Module):
    def forward(
        self, image: Float[torch.Tensor, "crops 3 244 244"]
    ) -> Float[torch.Tensor, "crops 1024"]:
        with torch.no_grad():
            return clip_model.encode_image(image).float()


class ClipFrozenTxtEnc(nn.Module):
    def forward(
        self, text: Int[torch.Tensor, "prompts 77"]
    ) -> Float[torch.Tensor, "prompts 1024"]:
        with torch.no_grad():
            return clip_model.encode_text(text).float()


# %%
clip_frozen_img_encoder: ClipFrozenImgEnc = ClipFrozenImgEnc()
clip_frozen_txt_encoder: ClipFrozenTxtEnc = ClipFrozenTxtEnc()

# %% [markdown]
# ### utils

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
def unzip(batch: list[tuple[T, ...]]) -> tuple[tuple[T, ...], ...]:
    """

    >>> unzip([('A', 1), ('B', 2)])
    (('A', 'B'), (1, 2))

    """
    return tuple(zip(*batch))


# %%
def best_bbox(
    pred: Float[torch.Tensor, "crops 4"], groundtruth: Float[torch.Tensor, "1 4"]
) -> int:
    """

    >>> best_bbox(
    ...     torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 2, 2]]),
    ...     torch.tensor([[0, 0, 1, 1]])
    ... )
    0

    >>> best_bbox(
    ...     torch.tensor([[0, 0, 0, 0], [0, 0, 2, 2], [1, 1, 2, 2]]),
    ...     torch.tensor([[0, 0, 1, 1]])
    ... )
    1

    """
    return torch.argmax(box_iou(pred, groundtruth)).item()


# %%
def eval_summary(model: nn.Module):
    summary(
        model,
        input_size=[(5, 3, 244, 244), (2, 77)],
        dtypes=[torch.float, torch.int],
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )


# %%
def contrastive_summary(model: nn.Module):
    summary(
        model,
        input_size=[(8, 3, 244, 244), (8, 77)],
        dtypes=[torch.float, torch.int],
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )


# %% [markdown]
# ### dataset

# %%
path_root: str = os.path.join("refcocog", "")
path_annotations: str = os.path.join(path_root, "annotations", "")
path_bboxes: str = os.path.join(path_root, "bboxes", "")
path_images: str = os.path.join(path_root, "images", "")

path_refs: str = os.path.join(path_annotations, "refs.csv")
path_sentences: str = os.path.join(path_annotations, "sentences.csv")

path_DETR: str = os.path.join(path_bboxes, "bboxes[DETR].csv")
path_YOLOv5: str = os.path.join(path_bboxes, "bboxes[YOLOv5].csv")
path_YOLOv8: str = os.path.join(path_bboxes, "bboxes[YOLOv8].csv")

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

# %%
TensorImage = UInt[torch.Tensor, "3 H W"]

# %%
class CocoDataset(Dataset[tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]]):
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
                    if bbox.confidence > .25  # lower bound on confidence
                ])
            ]
            for xyxy in [torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)])]
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
class Coco4MetricsDataset(Dataset[tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]]):

    def __init__(
        self,
        split: Split,
        img2bboxes: dict[str, list[BBox]],
        limit: int = -1,
    ):
        self.items: list[tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]] = [
            (xyxys, xyxy)
            for ref in refs
            if ref.split == split
            for bboxes in [img2bboxes[ref.file_name]]
            for xyxys in [torch.tensor([ (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.confidence) for bbox in bboxes ], dtype=torch.float)]
            for xyxy in [torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)], dtype=torch.float)]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]:
        return self.items[index]


# %%
class Coco4TrainingDataset(
    Dataset[
        tuple[
            list[TensorImage],
            list[str],
            int,
            Float[torch.Tensor, "crops 4"],
            Float[torch.Tensor, "1 4"],
        ]
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
                str,
                list[str],
                int,
                Float[torch.Tensor, "X 4"],
                Float[torch.Tensor, "1 4"],
            ]
        ] = [
            (img, sents, i, xyxys, xyxy)
            for ref in refs
            if ref.split == split
            for img in [os.path.join(path_images, ref.file_name)]
            for sents in [id2sents[ref.ref_id]]
            for bboxes in [img2bboxes[ref.file_name]]
            for xyxys in [
                torch.tensor([
                    (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
                    for bbox in bboxes
                    if bbox.confidence > .25  # lower bound on confidence
                    if bbox.xmax - bbox.xmin > 16  # lower bound on width
                    if bbox.ymax - bbox.ymin > 16  # lower bound on heigth
                ])
            ]
            if xyxys.shape[0] > 1 # lower bound on bbox per image
            for xyxy in [
                torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)])
            ]
            for ious in [box_iou(xyxys, xyxy)]
            if torch.max(ious).item() > .5  # ensure at least .5 of maximum IoU
            for i in [torch.argmax(ious).item()]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(
        self, index: int
    ) -> tuple[
        list[TensorImage],
        list[str],
        int,
        Float[torch.Tensor, "crops 4"],
        Float[torch.Tensor, "1 4"],
    ]:
        file_name, sents, i, xyxys, xyxy = self.items[index]
        img: TensorImage = read_image(file_name, ImageReadMode.RGB).to(device)

        xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt="xyxy", out_fmt="xywh").round().int()

        crops: list[TensorImage] = [
            crop(img, top=y, left=x, height=h, width=w)
            for xywh in xywhs
            for [x, y, w, h] in [xywh.tolist()]
        ]

        return crops, sents, i, xyxys, xyxy


# %%
class Coco4ContrastiveDataset(Dataset[tuple[TensorImage, list[str]]]):
    def __init__(
        self,
        split: Split,
        limit: int = -1,
    ):
        self.items: list[tuple[str, list[str], Float[torch.Tensor, "1 4"]]] = [
            (img, sents, xyxy)
            for ref in refs
            if ref.split == split
            for img in [os.path.join(path_images, ref.file_name)]
            for sents in [id2sents[ref.ref_id]]
            for xyxy in [torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)])]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[TensorImage, list[str]]:
        file_name, sents, xyxy = self.items[index]
        img: TensorImage = read_image(file_name, ImageReadMode.RGB).to(device)

        xywh: Int[torch.Tensor, "1 4"] = box_convert(xyxy, in_fmt="xyxy", out_fmt="xywh").round().int()
        [[x, y, w, h]] = xywh.tolist()

        return crop(img, top=y, left=x, height=h, width=w), sents


# %% [markdown]
# ### bbox model comparison

# %%
def metrics(dataset: Dataset[tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]]) -> pd.DataFrame:

    dataloader: DataLoader[tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]] = DataLoader(dataset, batch_size=None)
    Z: Float[torch.Tensor, '1 5'] = torch.zeros(1, 5)

    ious: list[float] = [ torch.max(box_iou(true_xyxy, torch.cat((Z, xyxys))[:, :4])).item() for xyxys, true_xyxy in tqdm(dataloader) ]
    rs: list[int] = [ xyxys.shape[0] for xyxys, _ in tqdm(dataloader) ]

    return pd.DataFrame({'iou': ious, '#': rs})


# %%
splits: list[Split] = ['train', 'val', 'test']
report: pd.DataFrame = pd.concat(
    [
        pd.concat(
            [yolov5, yolov8, detr],
            axis=1,
            keys=['yolov5', 'yolov8', 'detr']
        ).describe()
        for split in splits
        for yolov5 in [metrics(Coco4MetricsDataset(split, img2yolov5))]
        for yolov8 in [metrics(Coco4MetricsDataset(split, img2yolov8))]
        for detr in [metrics(Coco4MetricsDataset(split, img2detr))]
    ],
    axis=1,
    keys=splits
)

# %%
report


# %% [markdown]
# ### steps

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

        progress = tqdm(data_loader, desc="showtime")

        for iter, (img, prompts, xyxys, true_xyxy) in zip(it.count(1), progress):
            true_i: int = best_bbox(xyxys, true_xyxy)

            # from xyxys to crops
            xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt="xyxy", out_fmt="xywh").round().int()

            crops: list[TensorImage] = [
                crop(img, top=y, left=x, height=h, width=w)
                for xywh in xywhs
                for [x, y, w, h] in [xywh.tolist()]
            ]

            # forward pass
            model_output: Float[torch.Tensor, "crops"] = model(crops, prompts)

            # get index of the predicted bounding box to compute IoU accuracy
            pred_i: int = torch.argmax(model_output).item()

            # https://github.com/pytorch/pytorch/issues/65449
            writer.add_image_with_boxes(
                tag=f"{iter}: {' ¶ '.join(prompts)}",
                img_tensor=img,
                box_tensor=torch.stack((xyxys[pred_i], xyxys[true_i], true_xyxy.squeeze())),
                labels=["prediction", "best region proposal", "ground truth"],
                global_step=global_step,
            )


# %%
def eval_step(
    model: nn.Module,
    data_loader: DataLoader[tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]],
    img_preprocess: t.Callable[[TensorImage], Float[torch.Tensor, "3 244 244"]],
) -> pd.DataFrame:
    model.eval()

    ious: list[float] = []
    coss: list[float] = []
    euds: list[float] = []

    with torch.inference_mode():
        img: TensorImage
        prompts: list[str]
        xyxys: Float[torch.Tensor, "crops 4"]
        xyxy: Float[torch.Tensor, "4"]

        progress = tqdm(data_loader, desc="eval")

        for iter, (img, prompts, xyxys, true_xyxy) in enumerate(progress):

            if xyxys.shape[0] == 0:
                xyxys = torch.tensor((0, 0, img.shape[3], img.shape[2]))

            # from xyxys to crops
            xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt="xyxy", out_fmt="xywh").round().int()

            crops: list[TensorImage] = [
                crop(img, top=y, left=x, height=h, width=w)
                for xywh in xywhs
                for [x, y, w, h] in [xywh.tolist()]
            ]

            # from true_xyxy to true_crop
            true_xywh: Int[torch.Tensor, "1 4"] = box_convert(true_xyxy, in_fmt="xyxy", out_fmt="xywh").round().int()

            true_crop: TensorImage
            [true_crop] = [
                crop(img, top=y, left=x, height=h, width=w)
                for xywh in true_xywh
                for [x, y, w, h] in [xywh.tolist()]
            ]

            # forward pass
            model_output: Float[torch.Tensor, "crops"] = model(crops, prompts)

            # get index of the predicted bounding box to compute IoU accuracy
            pred_i: int = torch.argmax(model_output).item()

            # get predicted bounding
            pred_xyxy: Float[torch.Tensor, "1 4"] = xyxys[pred_i].unsqueeze(0)

            iou: float = box_iou(true_xyxy, pred_xyxy).item()
            ious.append(iou)

            true_z: Float[torch.Tensor, "1 1024"] = clip_frozen_img_encoder(img_preprocess(true_crop).unsqueeze(0))
            pred_z: Float[torch.Tensor, "1 1024"] = clip_frozen_img_encoder(img_preprocess(crops[pred_i]).unsqueeze(0))

            cos: float = torch.nn.functional.cosine_similarity(true_z, pred_z).item()
            coss.append(cos)

            eud: float = torch.cdist(true_z, pred_z, p=2).item()
            euds.append(eud)

        return pd.DataFrame(
            {
                "iou": ious,
                "cos similarity": coss,
                "euclidean distance": euds,
            }
        )

# %%
def compare(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mAP[IoU .3]": [(report["iou"] >= 0.3).sum() / report["iou"].count() for report in reports.values()],
            "mAP[IoU .5]": [(report["iou"] >= 0.5).sum() / report["iou"].count() for report in reports.values()],
            "mAP[IoU .7]": [(report["iou"] >= 0.7).sum() / report["iou"].count() for report in reports.values()],
            "mIoU": [report["iou"].mean() for report in reports.values()],
            "mCos": [report["cos similarity"].mean() for report in reports.values()],
            "mED": [report["euclidean distance"].mean() for report in reports.values()],
        },
        index=reports.keys(),
    )


# %% [markdown]
# ### baseline model

# %%
class ClipWrapper(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.img_preprocess: Compose = preprocess
        self.txt_preprocess: t.Callable[[t.Union[str, list[str]]], Float[torch.Tensor, "77"]] = clip.tokenize
        self.clip_model: CLIP = clip_model

    def forward(
        self, crops: list[TensorImage], prompts: list[str]
    ) -> Float[torch.Tensor, "crops 1"]:
        with torch.no_grad():
            # step 1: preprocess crops as required by the visual encoder
            crops_preprocessed: Float[torch.Tensor, "crops 3 244 244"] = torch.stack([
                self.img_preprocess(crop)
                for crop in crops
            ])

            # step 2: preprocess prompts as required by the text encoder
            prompts_preprocessed: Int[torch.Tensor, "prompts 77"] = self.txt_preprocess(prompts)

            similarity_matrix: Float[torch.Tensor, "prompts crops"]
            _, similarity_matrix = self.clip_model(
                crops_preprocessed,
                prompts_preprocessed,
            )

            return torch.mean(similarity_matrix, dim=0)


# %%
eval_summary(
    clip_model
)


# %% [markdown]
# ### standard fine tuning

# %%
class ClipSfCore(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,
        txt_encoder: nn.Module,
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder

    def cosine_similarity(
        self,
        crops_z: Float[torch.Tensor, "crops 1024"],
        prompts_z: Float[torch.Tensor, "prompts 1024"],
    ) -> Float[torch.Tensor, "prompts crops"]:
        # normalise the image and the text
        crops_z: Float[torch.Tensor, "crops 1024"] = crops_z / crops_z.norm(dim=-1, keepdim=True)
        prompts_z: Float[torch.Tensor, "prompts 1024"] = prompts_z / prompts_z.norm(dim=-1, keepdim=True)

        # evaluate the cosine similarity between the sets of features
        return prompts_z @ crops_z.T

    def forward(
        self,
        crops: Float[torch.Tensor, "crops 3 244 244"],
        prompts: Int[torch.Tensor, "prompts 77"],
    ) -> Float[torch.Tensor, "crops 1"]:
        # step 1: compute crop representation in the latent space
        crop_z: Float[torch.Tensor, "crops 1024"] = self.img_encoder(crops)

        # step 2: compute prompt representation in the latent space
        prompt_z: Int[torch.Tensor, "prompts 1024"] = self.txt_encoder(prompts)

        # step 3: evaluate logits
        similarity_matrix: Float[torch.Tensor, "prompts crops"] = self.cosine_similarity(crop_z, prompt_z)

        # step 4: crops classification
        return torch.mean(similarity_matrix, dim=0)


# %%
class ClipSf(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,
        txt_encoder: nn.Module,
    ):
        super().__init__()
        self.img_preprocess: Compose = preprocess
        self.txt_preprocess: t.Callable[[t.Union[str, list[str]]], Float[torch.Tensor, "77"]] = clip.tokenize
        self.core = ClipSfCore(img_encoder, txt_encoder)

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
eval_summary(
    ClipSf(
        img_encoder=nn.Sequential(
            clip_frozen_img_encoder,
            nn.ReLU(),
            nn.Linear(1024, 1024)
        ),
        txt_encoder=clip_frozen_txt_encoder,
    ).to(device).core
)

_ = lambda params: torch.optim.SGD(params=params, lr=.01, weight_decay=.01, momentum=.9)

# %%
eval_summary(
    ClipSf(
        img_encoder=nn.Sequential(
            clip_frozen_img_encoder,
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
        ),
        txt_encoder=nn.Sequential(
            clip_frozen_txt_encoder,
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
        ),
    ).to(device).core
)

_ = lambda params: torch.optim.SGD(params=params, lr=.01, weight_decay=.01, momentum=.9)

# %%
eval_summary(
    ClipSf(
        img_encoder=nn.Sequential(
            clip_frozen_img_encoder,
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
        ),
        txt_encoder=nn.Sequential(
            clip_frozen_txt_encoder,
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 1024),
        ),
    ).to(device).core
)

_ = lambda params: torch.optim.Adadelta(params=params, lr=.0015, weight_decay=.01)

# %%
eval_summary(
    ClipSf(
        img_encoder=nn.Sequential(
            clip_frozen_img_encoder,
            nn.ReLU(),
            nn.Linear(1024, 512),
        ),
        txt_encoder=nn.Sequential(
            clip_frozen_txt_encoder,
            nn.ReLU(),
            nn.Linear(1024, 512),
        ),
    ).to(device).core
)

_ = lambda params: torch.optim.SGD(params=params, lr=.01, weight_decay=.01, momentum=.9)

# %%
loss_fn: t.Callable[[Float[torch.Tensor, "crops"], Int[torch.Tensor, "1"]], Float[torch.Tensor, "1"]] = nn.functional.cross_entropy

# %%
def training_step(
        model: nn.Module,
        data_loader: DataLoader[
            tuple[
                tuple[TensorImage, ...],
                tuple[str, ...],
                int,
                Float[torch.Tensor, "crops 4"],
                Float[torch.Tensor, "1 4"],
            ]
        ],
        optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()

    running_loss: float = 0
    running_acc: float = 0
    progress = tqdm(data_loader, desc="training")

    cropss: tuple[tuple[TensorImage, ...], ...]
    promptss: tuple[tuple[str, ...], ...]
    true_is: tuple[int, ...]
    xyxyss: tuple[Float[torch.Tensor, "crops 4"], ...]
    true_xyxys: tuple[Float[torch.Tensor, "1 4"], ...]

    for iter, (cropss, promptss, true_is, xyxyss, true_xyxys) in zip(it.count(1), progress):
        # forward pass
        preds: list[Float[torch.Tensor, "crops"]] = [
            model(crops, prompts) for crops, prompts in zip(cropss, promptss)
        ]

        # calculate loss
        losses: Float[torch.Tensor, "batch"] = torch.stack([
            loss_fn(pred, torch.tensor(true_i))
            for pred, true_i in zip(preds, true_is)
        ])
        loss: Float[torch.Tensor, "1"] = torch.mean(losses)
        running_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate IoU accuracy
        with torch.inference_mode():
            # # get indexes of the predicted bounding box to compute IoU accuracy
            pred_is: list[int] = [
                torch.argmax(pred).item()
                for pred in preds
            ]

            # # get predicted bounding boxes
            pred_xyxys: list[Float[torch.Tensor, "4"]] = [
                xyxys[pred_i]
                for xyxys, pred_i in zip(xyxyss, pred_is)
            ]

            # # IoU
            acc: float = torch.mean(box_iou(torch.cat(true_xyxys), torch.stack(pred_xyxys)).diagonal()).item()
            running_acc += acc

            progress.set_postfix(
                {
                    "loss": running_loss / iter,
                    "iou": running_acc / iter,
                },
                refresh=False,
            )

    return running_loss / len(data_loader), running_acc / len(data_loader)


# %%
def test_step(
        model: nn.Module,
        data_loader: DataLoader[tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]],
) -> tuple[float, float]:
    model.eval()

    running_loss: float = 0
    running_acc: float = 0
    progress = tqdm(data_loader, desc="testing")

    with torch.inference_mode():
        img: TensorImage
        prompts: list[str]
        xyxys: Float[torch.Tensor, "crops 4"]
        xyxy: Float[torch.Tensor, "4"]

        for iter, (img, prompts, xyxys, true_xyxy) in zip(it.count(1), progress):
            true_i: int = best_bbox(xyxys, true_xyxy)

            # from xyxys to crops
            xywhs: Int[torch.Tensor, "X 4"] = box_convert(xyxys, in_fmt="xyxy", out_fmt="xywh").round().int()

            crops: list[TensorImage] = [
                crop(img, top=y, left=x, height=h, width=w)
                for xywh in xywhs
                for [x, y, w, h] in [xywh.tolist()]
            ]

            # forward pass
            model_output: Float[torch.Tensor, "crops"] = model(crops, prompts)

            # calculate loss
            loss: float = loss_fn(model_output, torch.tensor(true_i)).item()
            running_loss += loss

            # calculate IoU accuracy

            # # get index of the predicted bounding box to compute IoU accuracy
            pred_i: int = torch.argmax(model_output).item()

            # # get predicted bounding
            pred_xyxy: Float[torch.Tensor, "4"] = xyxys[pred_i]

            # # IoU
            acc: float = box_iou(true_xyxy, pred_xyxy.unsqueeze(0)).item()
            running_acc += acc

            progress.set_postfix(
                {
                    "loss": running_loss / iter,
                    "iou": running_acc / iter,
                },
                refresh=False,
            )

        return running_loss / len(data_loader), running_acc / len(data_loader)


# %%
keys: list[str] = [f"net{i + 1}" for i in range(4)]

# %%
pd.concat(
    [
        pd.read_csv(f"assets/standard-finetuning/eval-{key}.csv", index_col=0)
        for key in keys
    ],
    axis=1,
    keys=keys,
)


# %%
pd.read_csv("assets/standard-finetuning/comparing.csv", index_col=0)

# %% [markdown]
# ### FLYP

# %%
class ClipFlypCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = nn.Sequential(
            clip_frozen_img_encoder,
            nn.ReLU(),
            nn.Dropout(.25),
            nn.Linear(1024, 1024),
        )
        self.txt_encoder = nn.Sequential(
            clip_frozen_txt_encoder,
            nn.ReLU(),
            nn.Dropout(.25),
            nn.Linear(1024, 1024),
        )
        # the temperature parameter is added as suggested by the original paper in order to prevent training instability
        self.logit_scale: Float[torch.Tensor, "1"] = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07))
        )

    def forward(
        self,
        crop: Float[torch.Tensor, "entries 3 244 244"],
        prompt: Int[torch.Tensor, "entries 77"],
    ) -> tuple[
        Float[torch.Tensor, "1024"],
        Float[torch.Tensor, "1024"],
        Float[torch.Tensor, "entries"],
    ]:
        # step 1: compute crop representation in the latent space
        crop_z: Float[torch.Tensor, "entries 1024"] = self.img_encoder(crop)

        # step 2: compute prompt representation in the latent space
        prompt_z: Int[torch.Tensor, "entries 1024"] = self.txt_encoder(prompt)

        return crop_z, prompt_z, self.logit_scale.exp()


# %%
class ClipFlyp(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_preprocess: Compose = preprocess
        self.txt_preprocess: t.Callable[[t.Union[str, list[str]]], Float[torch.Tensor, "77"]] = clip.tokenize
        self.core = ClipFlypCore()

    def forward(
        self, entries: list[tuple[TensorImage, str]]
    ) -> Float[torch.Tensor, "entries"]:
        # step 1: preprocess crops as required by the visual encoder
        with torch.no_grad():
            crops_preprocessed: Float[torch.Tensor, "entries 3 244 244"] = torch.stack([
                self.img_preprocess(crop)
                for crop, _ in entries
            ])

        # step 2: preprocess prompts as required by the text encoder
        with torch.no_grad():
            prompts_preprocessed: Int[torch.Tensor, "entries 77"] = self.txt_preprocess([
                prompt
                for _, prompt in entries
            ])

        return self.core(crops_preprocessed, prompts_preprocessed)

# %%
contrastive_summary(
    ClipFlyp().to(device).core
)

_ = lambda params: torch.optim.SGD(params=params, lr=.01, weight_decay=.01, momentum=.9)
_ = lambda params: torch.optim.Adam(params=params, lr=.00043, weight_decay=.01)

# %%
class ClipFlypEvalCore(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,
        txt_encoder: nn.Module,
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder

    def cosine_similarity(
        self,
        crops_z: Float[torch.Tensor, "crops 1024"],
        prompts_z: Float[torch.Tensor, "prompts 1024"],
    ) -> Float[torch.Tensor, "prompts crops"]:
        # normalise the image and the text
        crops_z: Float[torch.Tensor, "crops 1024"] = crops_z / crops_z.norm(dim=-1, keepdim=True)
        prompts_z: Float[torch.Tensor, "prompts 1024"] = prompts_z / prompts_z.norm(dim=-1, keepdim=True)

        # evaluate the cosine similarity between the sets of features
        return prompts_z @ crops_z.T

    def forward(
        self,
        crops: Float[torch.Tensor, "crops 3 244 244"],
        prompts: Int[torch.Tensor, "prompts 77"],
    ) -> Float[torch.Tensor, "crops 1"]:
        # step 1: compute crop representation in the latent space
        crop_z: Float[torch.Tensor, "crops 1024"] = self.img_encoder(crops)

        # step 2: compute prompt representation in the latent space
        prompt_z: Int[torch.Tensor, "prompts 1024"] = self.txt_encoder(prompts)

        # step 3: evaluate logits
        similarity_matrix: Float[torch.Tensor, "prompts crops"] = self.cosine_similarity(crop_z, prompt_z)

        # step 4: crops classification
        return torch.mean(similarity_matrix, dim=0)

# %%
class ClipFlypEval(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,  # visual encoder
        txt_encoder: nn.Module,  # natural language prompts encoder
    ):
        super().__init__()
        self.img_preprocess: Compose = preprocess
        self.txt_preprocess: t.Callable[[t.Union[str, list[str]]], Float[torch.Tensor, "77"]] = clip.tokenize
        self.core = ClipFlypEvalCore(img_encoder, txt_encoder)

    def forward(
        self, crops: list[TensorImage], prompts: list[str]
    ) -> Float[torch.Tensor, "crops 1"]:
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
eval_summary(
    ClipFlypEval(
        ClipFlyp().core.img_encoder,
        ClipFlyp().core.txt_encoder,
    ).to(device).core
)

# %% [markdown]
# #### training

# %%
class ClipLoss(nn.Module):
    def forward(
        self,
        imgs_features: Float[torch.Tensor, "entries 1024"],
        txts_features: Float[torch.Tensor, "entries 1024"],
        logit_scale: Float[torch.Tensor, "1"],
    ) -> Float[torch.Tensor, "1"]:
        # compute logits per image and logits per text
        logits_per_image: Float[torch.Tensor, "entries entries"] = logit_scale * imgs_features @ txts_features.T
        logits_per_text: Float[torch.Tensor, "entries entries"] = logit_scale * txts_features @ imgs_features.T

        # get ground truth labels for the computation of the cross entropy loss
        labels: Int[torch.Tensor, "entries"] = torch.arange(logits_per_image.shape[0])

        return torch.stack((
            nn.functional.cross_entropy(logits_per_image, labels),
            nn.functional.cross_entropy(logits_per_text, labels),
        )).mean()


# %%
contrastive_loss_fn: t.Callable[
    [
        Float[torch.Tensor, "entries 1024"],
        Float[torch.Tensor, "entries 1024"],
        Float[torch.Tensor, "1"],
    ],
    Float[torch.Tensor, "1"],
] = ClipLoss()


# %%
def contrastive_training_step(
    model: ClipFlyp,
    data_loader: DataLoader[tuple[TensorImage, list[str]]],
    optimizer: torch.optim.Optimizer,
) -> float:
    running_loss: float = 0.0
    progress = tqdm(data_loader, desc="training")

    model.train()

    entries: tuple[tuple[TensorImage, list[str]], ...]
    entry: tuple[TensorImage, list[str]]

    for iter, entries in zip(it.count(1), progress):
        # forward computation
        imgs_features, txts_features, logit_scale = model(
            [(img, prompts[0]) for img, prompts in entries]
        )

        # calculate loss
        loss: Float[torch.Tensor, "1"] = contrastive_loss_fn(imgs_features, txts_features, logit_scale)
        running_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.core.logit_scale.clamp_(0, math.log(100))

            progress.set_postfix({"loss": running_loss / iter}, refresh=False)

    return running_loss / len(data_loader)


# %%
def contrastive_training_step_with_synonyms(
    model: ClipFlyp,
    data_loader: DataLoader[tuple[TensorImage, list[str]]],
    synonyms: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    running_loss: float = 0.0
    progress = tqdm(data_loader, desc="training")

    model.train()

    entries: list[list[tuple[TensorImage, str]]]
    entry: list[tuple[TensorImage, str]]

    for iter, entries in zip(it.count(1), progress):

        # forward computation
        imgs_features: Float[torch.Tensor, "entries*synonyms 1024"]
        txts_features: Float[torch.Tensor, "entries*synonyms 1024"]
        logit_scale: Float[torch.Tensor, "1"]
        imgs_features, txts_features, logit_scale = model(list(it.chain(*entries)))

        imgs_features_3d: Float[torch.Tensor, "entries synonyms 1024"] = imgs_features.view(len(entries), synonyms, 1024)
        imgs_features_3d: Float[torch.Tensor, "synonyms entries 1024"] = imgs_features_3d.transpose(0, 1)

        txts_features_3d: Float[torch.Tensor, "entries synonyms 1024"] = txts_features.view(len(entries), synonyms, 1024)
        txts_features_3d: Float[torch.Tensor, "synonyms entries 1024"] = txts_features_3d.transpose(0, 1)

        # calculate loss
        loss: Float[torch.Tensor, "1"] = torch.stack([
            loss_fn(imgs_features_2d, txts_features_2d, logit_scale)
            for imgs_features_2d, txts_features_2d in zip(imgs_features_3d, txts_features_3d)
        ]).mean()
        running_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.core.logit_scale.clamp_(0, math.log(100))

            progress.set_postfix({"loss": running_loss / iter}, refresh=False)

    return running_loss / len(data_loader)


# %%
def contrastive_test_step(
    model: ClipFlyp,
    data_loader: DataLoader[tuple[TensorImage, list[str]]],
) -> float:
    running_loss: float = 0.0
    progress = tqdm(data_loader, desc="testing")

    model.eval()

    with torch.inference_mode():
        entries: tuple[tuple[TensorImage, list[str]], ...]
        entry: tuple[TensorImage, list[str]]

        for iter, entries in zip(it.count(1), progress):
            # forward computation
            imgs_features, txts_features, logit_scale = model(
                [(img, prompts[0]) for img, prompts in entries]
            )

            # calculate loss
            loss: Float[torch.Tensor, "1"] = contrastive_loss_fn(imgs_features, txts_features, logit_scale)
            running_loss += loss.item()

            progress.set_postfix({"loss": running_loss / iter}, refresh=False)

    return running_loss / len(data_loader)


# %%
def contrastive_showtime(
    model: ClipFlyp,
    spli2loader: dict[Split, DataLoader[tuple[TensorImage, list[str]]]],
    writer: SummaryWriter,
    global_step: int,
) -> None:
    model.eval()

    with torch.inference_mode():

        for split, data_loader in spli2loader.items():

            progress = tqdm(data_loader, desc="showtime [{split}]")

            entries: tuple[tuple[TensorImage, list[str]], ...]
            entry: tuple[TensorImage, list[str]]

            for iter, entries in zip(it.count(1), progress):

                # forward computation
                imgs_features, txts_features, _ = model([
                    (img, prompts[0])
                    for img, prompts in entries
                ])

                imgs_features: Float[torch.Tensor, "entries 1024"] = imgs_features / imgs_features.norm(dim=-1, keepdim=True)
                txts_features: Float[torch.Tensor, "entries 1024"] = txts_features / txts_features.norm(dim=-1, keepdim=True)
                similarity: Float[torch.Tensor, "entries entries"] = (txts_features @ imgs_features.T).cpu()

                f: plt.Figure
                ax: plt.Axes
                f, ax = plt.subplots(1, 1, figsize=(10, 8))

                ax.imshow(similarity, vmin=torch.min(similarity).item(), vmax=torch.max(similarity).item())

                ax.set_yticks(
                    range(len(entries)),
                    ["\n".join(prompts) for _, prompts in entries],
                    fontsize=10,
                )
                ax.set_xticks([])

                for i, image in enumerate([ crop for crop, _ in entries ]):
                    ax.imshow(
                        image.permute(1, 2, 0).cpu(),
                        extent=(i - 0.5, i + 0.5, -1.6, -0.6),
                        origin="lower",
                    )

                for x in range(similarity.shape[1]):
                    for y in range(similarity.shape[0]):
                        ax.text(
                            x,
                            y,
                            f"{similarity[y, x]:.2f}",
                            ha="center",
                            va="center",
                            size=12,
                        )

                for side in ["left", "top", "right", "bottom"]:
                    f.gca().spines[side].set_visible(False)

                ax.set_xlim([-0.5, len(entries) - 0.5])
                ax.set_ylim([len(entries) + 0.5, -2])

                f.tight_layout()

                writer.add_figure(tag=f"matrix {iter}/{split}", figure=f, global_step=global_step)

# %%
keys: list[str] = ["flyp", "flyp-solve-overfitting", "flyp-optuna", "flyp-augmented"]


# %%
pd.concat(
    [
        pd.read_csv(f"assets/{key}/eval-FLYP-train.csv", index_col=0)
        for key in keys
    ],
    axis=1,
    keys=keys,
)

# %%
pd.concat(
    [
        pd.read_csv(f"assets/{key}/eval-FLYP-val.csv", index_col=0)
        for key in keys
    ],
    axis=1,
    keys=keys,
)

# %%
pd.concat(
    [
        pd.read_csv(f"assets/{key}/eval-FLYP-test.csv", index_col=0)
        for key in keys
    ],
    axis=1,
    keys=keys,
)

# %%
pd.read_csv("assets/comparing[train].csv", index_col=0)

# %%
pd.read_csv("assets/comparing[val].csv", index_col=0)

# %%
pd.read_csv("assets/comparing[test].csv", index_col=0)

# %% [markdown]
# Data augmentation

# %%
txt_transform: RandomChoice = RandomChoice([
    "A photo of {}".format,
    "A picture of {}".format,
    "An image of {}".format,
    "This is {}".format,
    "We can see {}".format,
])

# %%
img_transform: RandomChoice = RandomChoice([
    ColorJitter(brightness=0.5, hue=0.3), # randomly changes the brightness, saturation, and other properties of an image
    GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # performs gaussian blur transform on an image
    RandomPosterize(bits=2),  # randomly posterizes the image by reducing the number of bits of each color channel
    RandomSolarize(threshold=192.0),  # randomly solarizes the image by inverting all pixel values above the threshold
    RandomAdjustSharpness(sharpness_factor=2),  # randomly adjusts the sharpness of the given image
    RandomAutocontrast(),  # randomly applies autocontrast to the given image
    RandomEqualize(),  # randomly equalizes the histogram of the given image
    Grayscale(num_output_channels=3),  # converts an image to grayscale
])

# %%
BATCH_SIZE: int = 1024
SYNONYMS: int = 2
LIMIT: int = -1
EPOCHS: int = 50


# %%
def augment(batch: list[tuple[TensorImage, list[str]]]) -> list[list[tuple[TensorImage, str]]]:
    return [
        list(zip(
            random.sample(
                [img] +
                [img_transform(img) for _ in range(SYNONYMS - 1)],
                SYNONYMS
            ),
            random.sample(
                prompts +
                [txt_transform(random.choice(prompts)) for _ in range(SYNONYMS - len(prompts))],
                SYNONYMS
            )
        ))
        for img, prompts in batch
    ]


# %%
split2loader: dict[Split, DataLoader[tuple[TensorImage, list[str]]]] = {
    "train": DataLoader(
        dataset=Coco4ContrastiveDataset(split="train", limit=LIMIT),
        generator=g,
        batch_size=BATCH_SIZE,
        collate_fn=augment,
        shuffle=True,
    ),
    **{
        split: DataLoader(
            dataset=Coco4ContrastiveDataset(split=split, limit=LIMIT),
            batch_size=BATCH_SIZE,
            collate_fn=lambda x: x,
        )
        for split in ["val", "test"]
    }
}

split2showtime_dataloader: dict[Split, DataLoader[tuple[TensorImage, list[str]]]] = {
    split: DataLoader(
        dataset=Coco4ContrastiveDataset(split=split, limit=5 * 6),
        batch_size=6,
        collate_fn=lambda x: x,
        shuffle=False,
    )
    for split in ['train', 'val', 'test']
}


# %%
def training_loop(
        name: str,
        model: ClipFlyp,
        optimizer: t.Callable[[t.Iterable[torch.Tensor]], torch.optim.Optimizer],
) -> pd.DataFrame:
    loss: dict[str, list[float]] = defaultdict(list)

    # create a logger for the experiment
    with SummaryWriter(f"runs/{name}") as writer:
        # computes evaluation results before training
        print("Before training:")
        test_loss: float = contrastive_test_step(
            model=model,
            data_loader=split2loader["test"],
        )
        val_loss: float = contrastive_test_step(
            model=model,
            data_loader=split2loader["val"],
        )

        loss["test"].append(test_loss)
        loss["val"].append(val_loss)

        contrastive_showtime(
            model,
            split2showtime_dataloader,
            writer,
            0
        )

        # log to TensorBoard
        writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={
                "test": test_loss,
                "val": val_loss,
            },
            global_step=0,
        )

        progress = trange(EPOCHS, desc="EPOCHS")
        for epoch in progress:
            train_loss: float = contrastive_training_step_with_synonyms(
                model=model,
                data_loader=split2loader["train"],
                optimizer=optimizer(model.parameters()),
                synonyms=SYNONYMS
            )

            val_loss: float = contrastive_test_step(
                model=model,
                data_loader=split2loader["val"],
            )

            loss["train"].append(train_loss)
            loss["val"].append(val_loss)

            # log to TensorBoard
            writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict={
                    "train": train_loss,
                    "val": val_loss,
                },
                global_step=epoch + 1,
            )

            progress.set_postfix(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                },
                refresh=False,
            )

            # store model
            torch.save(obj=model.state_dict(), f=f"{name}-{(epoch + 1):02d}.pth")

        # compute final evaluation results
        print("After training:")

        test_loss: float = contrastive_test_step(
            model=model,
            data_loader=split2loader["test"],
        )

        loss["test"].append(test_loss)

        contrastive_showtime(
            model,
            split2showtime_dataloader,
            writer,
            EPOCHS
        )

        # log to TensorBoard
        writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={
                "test": test_loss,
            },
            global_step=EPOCHS,
        )

        return pd.concat(
            [
                pd.concat(
                    [pd.Series(v).describe() for v in loss.values()],
                    axis=1,
                    keys=[k for k in loss.keys()],
                ),
            ],
            axis=1,
            keys=["loss"],
        )

# %% [markdown]
# ### attention

# %%
class ClipContexCore(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,  # visual encoder
        txt_encoder: nn.Module,  # natural language prompts encoder
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=1)

    def contextualize(
        self,
        crops_z: Float[torch.Tensor, "crops 1024"],
        prompts_z: Float[torch.Tensor, "prompts 1024"],
    ) -> tuple[Float[torch.Tensor, "crops 1024"], Float[torch.Tensor, "prompts 1024"]]:
        # concatenate image embeedings and prompt embeedings in the same latent context
        concat: Float[torch.Tensor, "crops+prompts 1024"] = torch.cat((crops_z, prompts_z), dim=0)

        contextualized: Float[torch.Tensor, "crops+prompts 1024"]
        contextualized, _ = self.attention(concat, concat, concat)

        # retrive image_features and text_features by means of the previously stored indexes
        return contextualized[: crops_z.shape[0]], contextualized[-prompts_z.shape[0] :]

    def cosine_similarity(
        self,
        crops_z: Float[torch.Tensor, "crops 1024"],
        prompts_z: Float[torch.Tensor, "prompts 1024"],
    ) -> Float[torch.Tensor, "prompts crops"]:
        # normalise the image and the text
        crops_z: Float[torch.Tensor, "crops 1024"] = crops_z / crops_z.norm(dim=-1, keepdim=True)
        prompts_z: Float[torch.Tensor, "prompts 1024"] = prompts_z / prompts_z.norm(dim=-1, keepdim=True)

        # evaluate the cosine similarity between the sets of features
        return prompts_z @ crops_z.T

    def forward(
        self,
        crops: Float[torch.Tensor, "crops 3 244 244"],
        prompts: Int[torch.Tensor, "prompts 77"],
    ) -> Float[torch.Tensor, "crops 1"]:
        # step 1: compute crop representation in the latent space
        crops_z: Float[torch.Tensor, "crops 1024"] = self.img_encoder(crops)

        # step 2: compute prompt representation in the latent space
        prompts_z: Float[torch.Tensor, "prompts 1024"] = self.txt_encoder(prompts)

        # step 3: refine the latent representation of each text and image according to the overall context by means of the attention mechanism
        crop_context_z: Float[torch.Tensor, "crops 1024"]
        prompt_context_z: Float[torch.Tensor, "prompts 1024"]
        crop_context_z, prompt_context_z = self.contextualize(crops_z, prompts_z)

        # step 4: evaluate logits
        similarity_matrix: Float[torch.Tensor, "prompts crops"] = self.cosine_similarity(crop_context_z, prompt_context_z)

        # step 5: crops classification
        return torch.mean(similarity_matrix, dim=0)


# %%
class ClipContex(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,  # visual encoder
        txt_encoder: nn.Module,  # natural language prompts encoder
    ):
        super().__init__()
        self.core = ClipContexCore(img_encoder, txt_encoder)

        self.img_preprocess: Compose = preprocess
        self.txt_preprocess: t.Callable[[t.Union[str, list[str]]], Float[torch.Tensor, "77"]] = clip.tokenize

    def forward(
        self, crops: list[TensorImage], prompts: list[str]
    ) -> Float[torch.Tensor, "crops 1"]:
        # step 1: preprocess crops as required by the visual encoder
        with torch.no_grad():
            crops_preprocessed: Float[torch.Tensor, "crops 3 244 244"] = torch.stack([
                self.img_preprocess(crop)
                for crop in crops
            ])

        # step 2: preprocess prompts as required by the text encoder
        with torch.no_grad():
            prompts_preprocessed: Int[torch.Tensor, "prompts 77"] = self.txt_preprocess(
                prompts
            )

        return self.core(crops_preprocessed, prompts_preprocessed)

# %%
eval_summary(
    ClipContex(
        img_encoder=clip_frozen_img_encoder,
        txt_encoder=clip_frozen_txt_encoder,
    ).to(device).core
)
