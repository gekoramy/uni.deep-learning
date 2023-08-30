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
# %load_ext tensorboard

# %%
# %tensorboard --logdir ./runs

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
# #### Dataset and type declaration

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


class CocoDataset(
    Dataset[
        tuple[
            TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]
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
                    if bbox.xmax - bbox.xmin > 16  # lower bound on width
                    if bbox.ymax - bbox.ymin > 16  # lower bound on heigth
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

        xywh: Int[torch.Tensor, "1 4"] = box_convert(xyxy, in_fmt="xyxy", out_fmt="xywh").round().int()
        [[x, y, w, h]] = xywh.tolist()

        return crop(img, top=y, left=x, height=h, width=w), sents


# %% [markdown]
# # Fine tune like you pretrain
# In the following we try to fine tune CLIP image and text encoders using contrastive learning as proposed by the original paper.

# %%
clip_model, _ = clip.load("RN50", device=device)
clip_model.eval()

for p in clip_model.parameters():
    p.requires_grad = False


# %%
class ClipFreezedImgEnc(nn.Module):
    def forward(
        self, image: Float[torch.Tensor, "crops 3 244 244"]
    ) -> Float[torch.Tensor, "crops 1024"]:
        with torch.no_grad():
            return clip_model.encode_image(image).float()


class ClipFreezedTxtEnc(nn.Module):
    def forward(
        self, text: Int[torch.Tensor, "prompts 77"]
    ) -> Float[torch.Tensor, "prompts 1024"]:
        with torch.no_grad():
            return clip_model.encode_text(text).float()


# %%
clip_freezed_img_encoder: ClipFreezedImgEnc = ClipFreezedImgEnc()
clip_freezed_txt_encoder: ClipFreezedTxtEnc = ClipFreezedTxtEnc()


# %%
class ClipFlypCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = nn.Sequential(
            clip_freezed_img_encoder,
            nn.ReLU(),
            nn.Dropout(.25),
            nn.Linear(1024, 1024),
        )
        self.txt_encoder = nn.Sequential(
            clip_freezed_txt_encoder,
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
summary(
    ClipFlypCore(),
    input_size=[(8, 3, 244, 244), (8, 77)],
    dtypes=[torch.float, torch.int],
    col_names=["input_size", "output_size", "num_params", "trainable"],
)


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
loss_fn: t.Callable[
    [
        Float[torch.Tensor, "entries 1024"],
        Float[torch.Tensor, "entries 1024"],
        Float[torch.Tensor, "1"],
    ],
    Float[torch.Tensor, "1"],
] = ClipLoss()


# %%
def training_step(
    model: ClipFlyp,
    data_loader: DataLoader[tuple[TensorImage, list[str]]],
    synonyms: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    running_loss: float = 0.0
    progress = tqdm(data_loader, desc=f"training")

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
def test_step(
    model: ClipFlyp,
    data_loader: DataLoader[tuple[TensorImage, list[str]]],
) -> float:
    running_loss: float = 0.0
    progress = tqdm(data_loader, desc=f"testing")

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
            loss: Float[torch.Tensor, "1"] = loss_fn(imgs_features, txts_features, logit_scale)
            running_loss += loss.item()

            progress.set_postfix({"loss": running_loss / iter}, refresh=False)

    return running_loss / len(data_loader)


# %%
def training_showtime(
    model: ClipFlyp,
    spli2loader: dict[Split, DataLoader[tuple[TensorImage, list[str]]]],
    writer: SummaryWriter,
    global_step: int,
) -> None:
    model.eval()

    with torch.inference_mode():

        for split, data_loader in spli2loader.items():

            progress = tqdm(data_loader, desc=f"showtime [{split}]")

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


# %% [markdown]
# Data augmentation

# %%
# EDA
# paper: https://aclanthology.org/D19-1670.pdf
# paper: https://arxiv.org/abs/1907.03752
# code reference: https://github.com/dsfsi/textaugment
from textaugment import EDA

import nltk  # NLTK is a leading platform for building Python programs to work with human language data

nltk.download("stopwords")
nltk.download("wordnet")

eda: EDA = EDA(random_state=42)

# %%
# A large BART seq2seq (text2text generation) model fine-tuned on 3 paraphrase datasets.
# paper: https://arxiv.org/abs/1910.13461
# code reference: https://huggingface.co/eugenesiow/bart-paraphrase
from transformers import BartForConditionalGeneration, BartTokenizer

bart_model_name: str = "eugenesiow/bart-paraphrase"
bart_tokenizer: Compose = BartTokenizer.from_pretrained(bart_model_name)
bart_model: nn.Module = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)

bart_model.eval()

for p in bart_model.parameters():
    p.requires_grad = False

# %%
# PEGASUS fine-tuned for paraphrasing
# paper: https://arxiv.org/abs/1912.08777
# code reference: https://huggingface.co/tuner007/pegasus_paraphrase
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

pegasus_model_name: str = "tuner007/pegasus_paraphrase"
pegasus_tokenizer: Compose = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model: nn.Module = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(device)

pegasus_model.eval()

for p in pegasus_model.parameters():
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


def bart(txt: str) -> str:
    with torch.inference_mode():
        batch: dict[str, Int[torch.Tensor, "1 P"]] = bart_tokenizer(txt, return_tensors="pt")
        translated: Int[torch.Tensor, "1 X"] = bart_model.generate(batch["input_ids"])
        [out] = bart_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return out



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
img_transform: RandomChoice = RandomChoice([
    ColorJitter(brightness=0.5, hue=0.3),              # randomly changes the brightness, saturation, and other properties of an image
    GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # performs gaussian blur transform on an image
    RandomPosterize(bits=2),                           # randomly posterizes the image by reducing the number of bits of each color channel
    RandomSolarize(threshold=192.0),                   # randomly solarizes the image by inverting all pixel values above the threshold
    RandomAdjustSharpness(sharpness_factor=2),         # randomly adjusts the sharpness of the given image
    RandomAutocontrast(),                              # randomly applies autocontrast to the given image
    RandomEqualize(),                                  # randomly equalizes the histogram of the given image
    Grayscale(num_output_channels=3),                  # converts an image to grayscale
])

# %%
BATCH_SIZE: int = 256
SYNONYMS: int = 4
LIMIT: int = 2 * BATCH_SIZE  # (5_000 // BATCH_SIZE) * BATCH_SIZE
NUM_WORKERS: int = 0  # os.cpu_count() or 1
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
        dataset=Coco4ClipDataset(split="train", limit=LIMIT),
        generator=g,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=augment,
        shuffle=True,
    ),
    **{
        split: DataLoader(
            dataset=Coco4ClipDataset(split=split, limit=LIMIT),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=lambda x: x,
        )
        for split in ["val", "test"]
    }
}

split2showtime_dataloader: dict[Split, DataLoader[tuple[TensorImage, list[str]]]] = {
    split: DataLoader(
        dataset=Coco4ClipDataset(split=split, limit=5 * 6),
        batch_size=6,
        num_workers=NUM_WORKERS,
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
        test_loss = test_step(
            model=model,
            data_loader=split2loader["test"],
        )
        val_loss = test_step(
            model=model,
            data_loader=split2loader["val"],
        )

        loss["test"].append(test_loss)
        loss["val"].append(val_loss)

        training_showtime(
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
            train_loss = training_step(
                model=model,
                data_loader=split2loader["train"],
                optimizer=optimizer(model.parameters()),
                synonyms=SYNONYMS
            )

            val_loss = test_step(
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

        test_loss = test_step(
            model=model,
            data_loader=split2loader["test"],
        )

        loss["test"].append(test_loss)

        training_showtime(
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


# %%
# instantiate the network and move it to the chosen device
name: str = "FLYP"
model: ClipFlyp = ClipFlyp().to(device)

# %%
report: pd.DataFrame = training_loop(
    name=name,
    model=model,
    optimizer=lambda params: torch.optim.Adam(params=params, lr=.00043, weight_decay=.01),
)
report.to_csv(f"training-{name}.csv")


# %% [markdown]
# # Test the model on our down stream task
# In the following of the notebook we test the performance of the trained model on our objective task.

# %% [markdown]
# ## Evaluation code

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
def showtime(
    model: nn.Module,
    data_loader: DataLoader[
        tuple[
            TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]
        ]
    ],
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

            # get predicted bounding
            pred_xyxy: Float[torch.Tensor, "4"] = xyxys[pred_i]

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
    data_loader: DataLoader[
        tuple[
            TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]
        ]
    ],
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

        progress = tqdm(data_loader, desc=f"eval")

        for iter, (img, prompts, xyxys, true_xyxy) in enumerate(progress):
            true_i: int = best_bbox(xyxys, true_xyxy)

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

            true_z: Float[torch.Tensor, "1 1024"] = clip_freezed_img_encoder(img_preprocess(true_crop).unsqueeze(0))
            pred_z: Float[torch.Tensor, "1 1024"] = clip_freezed_img_encoder(img_preprocess(crops[pred_i]).unsqueeze(0))

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
            "mA[IoU .3]": [(report["iou"] >= 0.3).sum() / report["iou"].count() for report in reports.values()],
            "mA[IoU .5]": [(report["iou"] >= 0.5).sum() / report["iou"].count() for report in reports.values()],
            "mA[IoU .7]": [(report["iou"] >= 0.7).sum() / report["iou"].count() for report in reports.values()],
            "mIoU": [report["iou"].mean() for report in reports.values()],
            "mCos": [report["cos similarity"].mean() for report in reports.values()],
            "mED": [report["euclidean distance"].mean() for report in reports.values()],
        },
        index=reports.keys(),
    )


# %%
split2eval_loader: dict[
    Split,
    DataLoader[
        tuple[
            TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]
        ]
    ],
] = {
    split: DataLoader(
        dataset=CocoDataset(split=split, img2bboxes=img2detr, limit=LIMIT),
        batch_size=None,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )
    for split in ["train", "val", "test"]
}

showtime_eval_loader: DataLoader[
    tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]
] = DataLoader(
    dataset=CocoDataset(split="test", img2bboxes=img2detr, limit=20),
    batch_size=None,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

# %%
eval_model: ClipFlypEval = ClipFlypEval(
    model.core.img_encoder,
    model.core.txt_encoder,
)

# %%
summary(
    eval_model.core,
    input_size=[(5, 3, 244, 244), (2, 77)],
    dtypes=[torch.float, torch.int],
    col_names=["input_size", "output_size", "num_params", "trainable"],
)

# %%
eval_reports: dict[Split, pd.DataFrame] = {
    split: eval_step(eval_model, loader, preprocess)
    for split, loader in split2eval_loader.items()
}

# %%
for split, report in eval_reports.items():
    report.describe().to_csv(f"eval-{name}-{split}.csv")

# %%
compare(eval_reports).to_csv("comparing.csv")

# %%
with SummaryWriter(f"runs/{name}") as writer:
    showtime(eval_model, showtime_eval_loader, writer, EPOCHS)
