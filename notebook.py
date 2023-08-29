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
optuna
pandas
pydantic
regex
tensorboard
torch
torchinfo
torchvision
tqdm
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
import pickle

import clip
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import optuna

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
)
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm, trange
from optuna.trial import Trial
from optuna.study import Study

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
TensorImage = UInt[torch.Tensor, "3 H W"]


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
    def __init__(self, p: float):
        super().__init__()
        self.img_encoder = nn.Sequential(
            clip_freezed_img_encoder,
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, 1024),
        )
        self.txt_encoder = nn.Sequential(
            clip_freezed_txt_encoder,
            nn.ReLU(),
            nn.Dropout(p),
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
    ClipFlypCore(p=0.5),
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
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


preprocess: Compose = transform(224)


# %%
class ClipFlyp(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.img_preprocess: Compose = preprocess
        self.txt_preprocess: t.Callable[
            [t.Union[str, list[str]]], Float[torch.Tensor, "77"]
        ] = clip.tokenize
        self.core = ClipFlypCore(p=p)

    def forward(
        self, entries: list[tuple[TensorImage, str]]
    ) -> Float[torch.Tensor, "entries"]:
        # step 1: preprocess crops as required by the visual encoder
        with torch.no_grad():
            crops_preprocessed: Float[torch.Tensor, "entries 3 244 244"] = torch.stack([
                self.img_preprocess(crop) for crop, _ in entries
            ])

        # step 2: preprocess prompts as required by the text encoder
        with torch.no_grad():
            prompts_preprocessed: Int[torch.Tensor, "entries 77"] = self.txt_preprocess([
                prompt for _, prompt in entries
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

        return torch.stack(
            (
                nn.functional.cross_entropy(logits_per_image, labels),
                nn.functional.cross_entropy(logits_per_text, labels),
            )
        ).mean()


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
    optimizer: torch.optim.Optimizer,
) -> float:
    running_loss: float = 0.0

    model.train()

    entries: tuple[tuple[TensorImage, list[str]], ...]
    entry: tuple[TensorImage, list[str]]

    for entries in data_loader:
        # forward computation
        imgs_features, txts_features, logit_scale = model(
            [(img, prompts[0]) for img, prompts in entries]
        )

        # calculate loss
        loss: Float[torch.Tensor, "1"] = loss_fn(imgs_features, txts_features, logit_scale)
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

    return running_loss / len(data_loader)


# %%
def test_step(
    model: ClipFlyp,
    data_loader: DataLoader[tuple[TensorImage, list[str]]],
) -> float:
    running_loss: float = 0.0

    model.eval()

    with torch.inference_mode():
        entries: tuple[tuple[TensorImage, list[str]], ...]
        entry: tuple[TensorImage, list[str]]

        for entries in data_loader:
            # forward computation
            imgs_features, txts_features, logit_scale = model(
                [(img, prompts[0]) for img, prompts in entries]
            )

            # calculate loss
            loss: Float[torch.Tensor, "1"] = loss_fn(imgs_features, txts_features, logit_scale)
            running_loss += loss.item()

    return running_loss / len(data_loader)


# %%
BATCH_SIZE: int = 1024
LIMIT: int = 30 * BATCH_SIZE
EPOCHS: int = 10

# %%
split2loader: dict[Split, DataLoader[tuple[TensorImage, list[str]]]] = {
    split: DataLoader(
        dataset=Coco4ClipDataset(split=split, limit=LIMIT),
        generator=g,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: x,
        shuffle=(split == "train"),
    )
    for split in ["train", "val"]
}


# %%
def objective(trial: Trial):

    lr: float = trial.suggest_float("learning rate", 1e-5, .1, log=True)
    p: float = trial.suggest_float("dropout", .1, .9)
    optim: t.Literal["Adam", "SGD"] = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    optuna_model: ClipFlyp = ClipFlyp(p=p).to(device)

    match optim:
      case "Adam":
            optimizer: torch.optim.Optimizer = torch.optim.Adam(
                params=optuna_model.parameters(),
                lr=lr,
                weight_decay=.01
            )

      case "SGD":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                params=optuna_model.parameters(),
                lr=lr,
                weight_decay=.01,
                momentum=.9
            )

    for epoch in trange(EPOCHS):

        training_step(
            model = optuna_model,
            data_loader = split2loader["train"],
            optimizer = optimizer,
        )

        val_loss = test_step(
            model = optuna_model,
            data_loader = split2loader["val"],
        )

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


# %%
optuna.logging.set_verbosity(optuna.logging.WARNING)

study: optuna.study.Study = optuna.create_study(
    study_name="optuna-hyperparameter-optimization",
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(),
    load_if_exists=False,
)

study.optimize(
    func=objective,
    timeout=5 * 60 * 60,
    show_progress_bar=True,
)

# %%
with open("study.pickle", "wb") as f:
    pickle.dump(study, f)
