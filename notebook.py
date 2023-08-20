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
jaxtyping
matplotlib
more-itertools
pandas
pydantic
torch
torchvision
tqdm
END

pip install -q -r requirements.txt

# %%
import PIL.Image
import csv
import itertools as it
import os
import pandas as pd
import torch
import torchvision
import typing as t

from PIL.Image import Image
from collections import defaultdict
from jaxtyping import Float, UInt
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from tqdm import tqdm

# %%
path_root: str = os.path.join('dataset', 'refcocog', '')
path_annotations: str = os.path.join(path_root, 'annotations', '')
path_bboxes: str = os.path.join(path_root, 'bboxes', '')
path_images: str = os.path.join(path_root, 'images', '')

path_refs: str = os.path.join(path_annotations, 'refs.csv')
path_sentences: str = os.path.join(path_annotations, 'sentences.csv')

path_DETR: str = os.path.join(path_bboxes, 'bboxes[DETR].csv')
path_YOLOv5: str = os.path.join(path_bboxes, 'bboxes[YOLOv5].csv')
path_YOLOv8: str = os.path.join(path_bboxes, 'bboxes[YOLOv8].csv')

# %%
Split = t.Literal['train', 'test', 'val']

@dataclass
class Ref:
    ref_id: int  # unique id for refering expression
    file_name: str  # file name of image relative to img_root
    split: Split
    xmin: float
    ymin: float
    xmax: float
    ymax: float


with open(path_refs, 'r') as f:
    raw = csv.DictReader(f)
    refs: list[Ref] = [ Ref(**row) for row in raw ]

# %%
T = t.TypeVar('T')
K = t.TypeVar('K')
V = t.TypeVar('V')

def groupby(
    xs: list[T],
    map_key: t.Callable[[T], K],
    map_value: t.Callable[[T], V] = lambda x: x
) -> dict[K, list[V]]:
    return {
        k: [ map_value(v) for v in vs ]
        for k, vs in it.groupby(sorted(xs, key=map_key), key=map_key)
    }


# %%
@dataclass
class Sentence:
    ref_id: int  # unique id for refering expression
    sent: str


with open(path_sentences, 'r') as f:
    raw = csv.DictReader(f)
    sentences: list[Sentence] = [ Sentence(**row) for row in raw ]


id2sents: dict[int, list[str]] = groupby(sentences, lambda x: x.ref_id, lambda x: x.sent)


# %%
@dataclass
class BBox:
    file_name: str  # file name of image relative to img_root
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float


with open(path_DETR, 'r') as f:
    raw = csv.DictReader(f)
    bboxes: list[BBox] = [ BBox(**row) for row in raw ]

img2detr: dict[str, list[BBox]] = defaultdict(list, groupby(bboxes, lambda x: x.file_name))


with open(path_YOLOv5, 'r') as f:
    raw = csv.DictReader(f)
    bboxes: list[BBox] = [ BBox(**row) for row in raw ]

img2yolov5: dict[str, list[BBox]] = defaultdict(list, groupby(bboxes, lambda x: x.file_name))


with open(path_YOLOv8, 'r') as f:
    raw = csv.DictReader(f)
    bboxes: list[BBox] = [ BBox(**row) for row in raw ]

img2yolov8: dict[str, list[BBox]] = defaultdict(list, groupby(bboxes, lambda x: x.file_name))


# %%
class CocoMetricsDataset(Dataset[tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]]):

    def __init__(
        self,
        split: Split,
        img2bboxes: dict[str, list[BBox]],
        limit: int = -1,
    ):
        self.__init__
        self.items: list[tuple[Float[torch.Tensor, 'X 5'], Float[torch.Tensor, '1 4']]] = [
            (xyxys, xyxy)
            for ref in refs
            if ref.split == split
            for img in [os.path.join(path_images, ref.file_name)]
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
        for yolov5 in [metrics(CocoMetricsDataset(split, img2yolov5))]
        for yolov8 in [metrics(CocoMetricsDataset(split, img2yolov8))]
        for detr in [metrics(CocoMetricsDataset(split, img2detr))]
    ],
    axis=1,
    keys=splits
)

# %%
display(report)
