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
pandas
pydantic
regex
sentencepiece
tensorboard
textaugment
torch
torchinfo
torchvision
tqdm
transformers
END

pip install -q -r requirements.txt
pip install -q git+https://github.com/openai/CLIP.git

# %%
import csv
import doctest
import itertools as it
import math
import os
import typing as t
import random

import clip
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from collections import defaultdict
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
    RandomInvert,
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
# setting a manual seed allow us to provide reprudicible results in this notebook
# https://pytorch.org/docs/stable/notes/randomness.html

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(False)  # CLIP uses non-deterministic algorithms
g: torch.Generator = torch.Generator(device=device).manual_seed(42)
random.seed(42)

# %% [markdown]
# ## Dataset preparation

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
TensorImage = UInt[torch.Tensor, "3 H W"]


class Coco4ClipDataset(Dataset[tuple[TensorImage, list[str]]]):
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

        xywh: Int[torch.Tensor, "1 4"] = (
            box_convert(xyxy, in_fmt="xyxy", out_fmt="xywh").round().int()
        )
        [[x, y, w, h]] = xywh.tolist()

        return crop(img, top=y, left=x, height=h, width=w), sents


# %% [markdown]
# # Image augmentation

# %%
transform: RandomChoice = RandomChoice([
    ColorJitter(brightness=0.5, hue=0.3),  # randomly changes the brightness, saturation, and other properties of an image
    GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # performs gaussian blur transform on an image
    RandomInvert(),  # randomly inverts the colors of the given image
    RandomPosterize(bits=2),  # randomly posterizes the image by reducing the number of bits of each color channel
    RandomSolarize(threshold=192.0),  # randomly solarizes the image by inverting all pixel values above the threshold
    RandomAdjustSharpness(sharpness_factor=2),  # randomly adjusts the sharpness of the given image
    RandomAutocontrast(),  # randomly applies autocontrast to the given image
    RandomEqualize(),  # randomly equalizes the histogram of the given image
    Grayscale(num_output_channels=3),  # converts an image to grayscale
])

# %%
for _, (c, prompt) in zip(range(5), Coco4ClipDataset("test")):
    f: plt.Figure
    axs: tuple[plt.Axes, plt.Axes]
    f, axs = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3))

    f.suptitle(prompt)
    for ax, img in zip(axs, [c, transform(c)]):
        ax.imshow(img.permute(1, 2, 0).cpu())
        ax.axis("off")

    f.tight_layout()
    plt.show()

# %% [markdown]
# # Text augmentation

# %%
# EDA
# paper: https://aclanthology.org/D19-1670.pdf
# paper: https://arxiv.org/abs/1907.03752
# code reference: https://github.com/dsfsi/textaugment
from textaugment import EDA

import nltk  # NLTK is a leading platform for building Python programs to work with human language data

nltk.download("stopwords")
nltk.download("wordnet")

# %%
# PEGASUS fine-tuned for paraphrasing
# paper: https://arxiv.org/abs/1912.08777
# code reference: https://huggingface.co/tuner007/pegasus_paraphrase
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

pegasus_model_name = "tuner007/pegasus_paraphrase"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(device)

pegasus_model.eval()

for p in pegasus_model.parameters():
    p.requires_grad = False

# %%
# A large BART seq2seq (text2text generation) model fine-tuned on 3 paraphrase datasets.
# paper: https://arxiv.org/abs/1910.13461
# code reference: https://huggingface.co/eugenesiow/bart-paraphrase
from transformers import BartForConditionalGeneration, BartTokenizer

bart_model_name = "eugenesiow/bart-paraphrase"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)

bart_model.eval()

for p in bart_model.parameters():
    p.requires_grad = False


# %%
def pegasus(txt: str) -> str:
    with torch.inference_mode():
        batch: dict[str, Int[torch.Tensor, "1 P"]] = pegasus_tokenizer(
            [txt],
            truncation=True,
            padding="longest",
            max_length=60,
            return_tensors="pt",
        )
        translated: Int[torch.Tensor, "1 X"] = pegasus_model.generate(
            **batch,
            max_length=60,
            num_beams=10,
            num_return_sequences=1,
            temperature=1.5
        )
        [out] = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return out


# %%
def bart(txt: str) -> str:
    with torch.inference_mode():
        batch: dict[str, Int[torch.Tensor, "1 P"]] = bart_tokenizer(txt, return_tensors="pt")
        translated: Int[torch.Tensor, "1 X"] = bart_model.generate(batch["input_ids"])
        [out] = bart_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return out


# %%
eda: EDA = EDA(random_state=42)

# %%
txt_transform: RandomChoice = RandomChoice([
    "A photo of {}".format,
    "A picture of {}".format,
    "An image of {}".format,
    "This is {}".format,
    "We can see {}".format,
    eda.synonym_replacement,
    pegasus,
    bart,
])

# %%
for _, (c, prompts) in zip(range(5), Coco4ClipDataset("test")):
    for prompt in prompts:
        print(prompt, f"\n~> {txt_transform(prompt)}")
