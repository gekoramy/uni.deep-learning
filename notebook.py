# %%
# %%shell
tee requirements.txt << END
jaxtyping
matplotlib
pandas
pydantic
torch
torchvision
tqdm
ultralytics
END

pip install -q -r requirements.txt

# %%
import json
import logging
import os
import pickle
import re
import torch
import torchvision
import PIL
import itertools as it
import pandas as pd

from datetime import datetime
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm
from typing import Literal, Callable, Mapping, TypeVar, Iterator, Iterable
from ultralytics import YOLO

# %% [markdown]
# Download the dataset

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
    # segmentation: list[list[float]]
    area: float  # number of pixel of the described object
    iscrowd: Literal[1, 0]  # Crowd annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people)
    image_id: int  # id of the target image
    bbox: tuple[float, float, float, float]  # bounding box coordinates [xmin, ymin, width, height]
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

id2annotation: Mapping[int, Annotation] = {
    x.id: x for x in Instances(**raw).annotations
}


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
            for xywh in [torch.tensor(id2annotation[ref.ann_id].bbox, dtype=torch.float)]
        ]
        self.len: int = len(self.items) if limit < 0 else min(limit, len(self.items))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]:
        i, ps, b = self.items[index]
        with PIL.Image.open(i) as img:
            img.load()
            return img, ps, b


# %%
dataloader: DataLoader[tuple[list[PIL.Image], list[list[str]], list[Float[torch.Tensor, "4"]]]] = DataLoader(
    dataset=CocoDataset(split="test", limit=100),
    batch_size=None,
)

# %%
device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# %%
def metrics(
    bbox_model: Callable[[PIL.Image, list[str]], Float[torch.Tensor, "X 4"]]
) -> pd.DataFrame:
    ious: list[float] = []
    rs: list[int] = []

    with torch.inference_mode():
        img: PIL.Image
        prompts: list[str]
        true_xywh: Float[torch.Tensor, "4"]

        for img, prompts, true_xywh in tqdm(dataloader):
            true_xyxy: Float[torch.Tensor, "1 4"] = torchvision.ops.box_convert(true_xywh.unsqueeze(0), in_fmt="xywh", out_fmt="xyxy").to(device)
            pred_xyxy: Float[torch.Tensor, "X 4"] = bbox_model(img, prompts)

            iou: float = torch.max(box_iou(true_xyxy, pred_xyxy)).item()
            r: int = pred_xyxy.shape[0]

            ious.append(iou)
            rs.append(r)

    return pd.DataFrame({"iou": ious, "#": rs})


# %%
yolo_v5_model = torch.hub.load("ultralytics/yolov5", "yolov5s", device=device, _verbose=False)

# %%
yolo_v8_model: YOLO = YOLO("yolov8s.pt")

# %%
Z: Float[torch.Tensor, "1 4"] = torch.zeros(1, 4).to(device)

# %%
yolo_v5_metrics: pd.DataFrame = metrics(
    lambda img, _: torch.cat(
        [Z] + [
            box[:, :4]
            for box in yolo_v5_model(img).xyxy
        ],
        0
    )
)

yolo_v5_metrics.describe()

# %%
yolo_v8_metrics: pd.DataFrame = metrics(
    lambda img, _: torch.cat(
        [Z] + [
            box.xyxy
            for pred in yolo_v8_model.predict(img, verbose=False)
            for box in pred.boxes
        ],
        0,
    )
)

yolo_v8_metrics.describe()
