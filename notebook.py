# %%
# %%shell
if ! [ -d dataset ]; then
  mkdir dataset &&
  gdown 1P8a1g76lDJ8cMIXjNDdboaRR5-HsVmUb &&
  tar -xf refcocog.tar.gz -C dataset &&
  rm refcocog.tar.gz
fi

# %%
# %%shell
tee requirements.txt << END
jaxtyping
matplotlib
pandas
pydantic
timm
torch
torchvision
tqdm
transformers
ultralytics
more-itertools
END

pip install -q -r requirements.txt
pip install -q -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

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
import more_itertools as mit

from datetime import datetime
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torchvision.io import read_image
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm
from typing import Literal, Callable, Mapping, TypeVar, Iterator, Iterable
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
from csv import writer

# %%
root = os.path.join("dataset", "refcocog", "")
data_instances = os.path.join(root, "annotations", "instances.json")
data_refs = os.path.join(root, "annotations", "refs(umd).p")
data_images = os.path.join(root, "images", "")

# %%
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


# %%
with open(data_instances, "r") as f:
    raw = json.load(f)

instances: Instances = Instances(**raw)

# %%
device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# %%
def store_bboxes(
    name_model: str, bbox_model: Callable[[PIL.Image], Float[torch.Tensor, "X 5"]]
) -> None:

    with open(f"bboxes[{name_model}].csv", "a") as f:
        wr = writer(f)
        wr.writerow(["file_name", "xmin", "ymin", "xmax", "ymax", "confidence"])

        with torch.inference_mode():
            img: Image

            for img in tqdm(instances.images):

                pil: PIL.Image = PIL.Image.open(os.path.join(data_images, img.file_name)).convert("RGB")
                bbox: Float[torch.Tensor, "X 5"] = bbox_model(pil).cpu()
                wr.writerow([img.file_name] + bbox.tolist())


# %%
CONFIDENCE: float = 0.1

# %%
yolo_v5_model = torch.hub.load("ultralytics/yolov5", "yolov5s", device=device, _verbose=False)
yolo_v5_model.conf = CONFIDENCE
yolo_v5_model.eval()
pass

# %%
yolo_v8_model: YOLO = YOLO("yolov8x.pt")
yolo_v8_model.to(device)

# %%
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
detr_model.eval()
pass

# %%
store_bboxes(
    "YOLOv5",
    lambda img: torch.cat(
        [
            box[:, :5]
            for box in yolo_v5_model(img).xyxy
        ]
    )
)

# %%
store_bboxes(
    "YOLOv8",
    lambda img: torch.cat(
        [
            pred.boxes.data[:, :5]
            for pred in yolo_v8_model.predict(img, verbose=False, conf=CONFIDENCE)
        ]
    )
)


# %%
def detr_with_conf(image: PIL.Image) -> Float[torch.Tensor, "X 5"]:
    inputs = detr_processor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = detr_model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > CONFIDENCE
    target_sizes = torch.tensor([image.size[::-1]])

    return torch.cat(
        [
            torch.cat((pred["boxes"], pred["scores"].unsqueeze(1)), 1)
            for pred in detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=CONFIDENCE)
        ]
    )


store_bboxes("DETR", detr_with_conf)
