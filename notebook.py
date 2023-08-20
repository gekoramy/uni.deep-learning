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
more-itertools
pydantic
regex
torch
torchvision
END

pip install -q -r requirements.txt

# %%
import json
import os
import pickle
import re
import torch
import itertools as it
import more_itertools as mit
import csv

from datetime import datetime
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torchvision.ops import box_convert
from typing import Literal, Callable, Mapping, TypeVar

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

instances: Instances = Instances(**raw)

# %%
true_xywh: Float[torch.Tensor, 'X 4'] = torch.tensor([ x.bbox for x in instances.annotations ])
true_xyxy: Float[torch.Tensor, 'X 4'] = box_convert(true_xywh, in_fmt='xywh', out_fmt='xyxy')
id2xyxy: dict[int, Float[torch.Tensor, 'X 4']] = dict(zip([ x.id for x in instances.annotations ], true_xyxy.tolist()))

# %%
with open('refs.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(('ref_id', 'file_name', 'split', 'xmin', 'xmax', 'ymin', 'ymax'))
    writer.writerows([ tuple((ref.ref_id, ref.file_name, ref.split, *id2xyxy[ref.ann_id])) for ref in refs ])

# %%
with open('sentences.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(('ref_id', 'sent'))
    writer.writerows(it.chain(*[ [ (ref.ref_id, sentence.sent) for sentence in ref.sentences ] for ref in refs ]))

# %%
# %%shell
if ! [ -d refcocog ]; then
    mkdir refcocog
    mkdir -p refcocog/annotations
    mkdir -p refcocog/bboxes
    cp -r dataset/refcocog/images refcocog
    cp refs.csv refcocog/annotations
    cp sentences.csv refcocog/annotations
    cp bboxes[DETR].csv refcocog/bboxes
    cp bboxes[YOLOv5].csv refcocog/bboxes
    cp bboxes[YOLOv8].csv refcocog/bboxes
fi

# %%
# %%shell
if ! [ -f refcocog.tar ]; then
    tar cf refcocog.tar refcocog
fi
