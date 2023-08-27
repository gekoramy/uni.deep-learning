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
tensorboard
torch
torchinfo
torchvision
tqdm
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
import os
import typing as t

import clip
import pandas as pd
import torch
import torch.nn as nn

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
# ### Utils

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
                    if bbox.confidence > 0.25  # lower bound on confidence
                    if bbox.xmax - bbox.xmin > 16  # lower limit on width
                    if bbox.ymax - bbox.ymin > 16  # lower limit on heigth
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
class CocoTrainDataset(
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
                    if bbox.confidence > 0.25  # lower bound on confidence
                    if bbox.xmax - bbox.xmin > 16  # lower limit on width
                    if bbox.ymax - bbox.ymin > 16  # lower limit on heigth
                ])
            ]
            if xyxys.shape[1] > 1  # lower bound on # of bboxes per image
            for xyxy in [torch.tensor([(ref.xmin, ref.ymin, ref.xmax, ref.ymax)])]
            for ious in [box_iou(xyxys, xyxy)]
            if torch.max(ious).item() > 0.5  # lower bound on maximum IoU
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


# %% [markdown]
# # Attention is all you need
# In the following of this notebook we try to fine tune CLIP using a self-attention based approach. In this context, we try to refine the latent representations of both visual and textual prompts by means of single head attention mechanism.

# %%
clip_model, _ = clip.load("RN50", device=device)
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
with SummaryWriter() as writer:
    writer.add_graph(
        model=ClipContexCore(
            img_encoder=clip_freezed_img_encoder,
            txt_encoder=clip_freezed_txt_encoder,
        ),
        input_to_model=[
            torch.ones((5, 3, 244, 244)),
            torch.ones((2, 77), dtype=torch.int),
        ],
    )

# %%
summary(
    ClipContexCore(
        img_encoder=clip_freezed_img_encoder,
        txt_encoder=clip_freezed_txt_encoder,
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
loss_fn: t.Callable[
    [Float[torch.Tensor, "crops"], Int[torch.Tensor, "1"]], Float[torch.Tensor, "1"]
] = nn.functional.cross_entropy


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

    for iter, (cropss, promptss, true_is, xyxyss, true_xyxys) in zip(
        it.count(1), progress
    ):
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
            pred_is: list[int] = [torch.argmax(pred).item() for pred in preds]

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
    data_loader: DataLoader[
        tuple[
            TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]
        ]
    ],
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

            # # IoU
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
                box_tensor=torch.stack(
                    (xyxys[pred_i], xyxys[true_i], true_xyxy.squeeze())
                ),
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

            true_z: Float[torch.Tensor, "1 1024"] = clip_freezed_img_encoder(
                img_preprocess(true_crop).unsqueeze(0)
            )
            pred_z: Float[torch.Tensor, "1 1024"] = clip_freezed_img_encoder(
                img_preprocess(crops[pred_i]).unsqueeze(0)
            )

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


# %% [markdown]
# #### main training-evaluation loop

# %%
BATCH_SIZE: int = 4
LIMIT: int = 1 * BATCH_SIZE
NUM_WORKERS: int = 0  # os.cpu_count() or 1
EPOCHS: int = 50

# %%
train_loader: DataLoader[
    tuple[
        tuple[TensorImage, ...],
        tuple[str, ...],
        int,
        Float[torch.Tensor, "crops 4"],
        Float[torch.Tensor, "1 4"],
    ]
] = DataLoader(
    dataset=CocoTrainDataset(split="train", img2bboxes=img2detr, limit=LIMIT),
    generator=g,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=unzip,
    shuffle=True,
)


split2loader: dict[
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


showtime_loader: DataLoader[
    tuple[TensorImage, list[str], Float[torch.Tensor, "X 4"], Float[torch.Tensor, "4"]]
] = DataLoader(
    dataset=CocoDataset(split="test", img2bboxes=img2detr, limit=20),
    batch_size=None,
    num_workers=NUM_WORKERS,
    shuffle=False,
)


# %%
def training_loop(
    name: str,
    model: nn.Module,
    optimizer: t.Callable[[t.Iterable[torch.Tensor]], torch.optim.Optimizer],
) -> pd.DataFrame:
    loss: dict[str, list[float]] = defaultdict(list)
    accs: dict[str, list[float]] = defaultdict(list)

    # create a logger for the experiment
    with SummaryWriter(f"runs/{name}") as writer:
        # computes evaluation results before training
        print("Before training:")
        bna_train_loss, bna_train_accuracy = test_step(
            model=model,
            data_loader=split2loader["train"],
        )
        test_loss, test_accuracy = test_step(
            model=model,
            data_loader=split2loader["test"],
        )
        val_loss, val_accuracy = test_step(
            model=model,
            data_loader=split2loader["val"],
        )

        showtime(model=model, data_loader=showtime_loader, writer=writer, global_step=0)

        loss["BA train"].append(bna_train_loss)
        accs["BA train"].append(bna_train_accuracy)
        loss["test"].append(test_loss)
        accs["test"].append(test_accuracy)
        loss["val"].append(val_loss)
        accs["val"].append(val_accuracy)

        # log to TensorBoard
        writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={
                "BA train": bna_train_loss,
                "test": test_loss,
                "val": val_loss,
            },
            global_step=0,
        )
        writer.add_scalars(
            main_tag="accuracy",
            tag_scalar_dict={
                "BA train": bna_train_accuracy,
                "test": test_accuracy,
                "val": val_accuracy,
            },
            global_step=0,
        )

        progress = trange(EPOCHS, desc="EPOCHS")
        for epoch in progress:
            train_loss, train_accuracy = training_step(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer(model.parameters()),
            )

            val_loss, val_accuracy = test_step(
                model=model,
                data_loader=split2loader["val"],
            )

            loss["train"].append(train_loss)
            accs["train"].append(train_accuracy)
            loss["val"].append(val_loss)
            accs["val"].append(val_accuracy)

            # log to TensorBoard
            writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict={
                    "train": train_loss,
                    "val": val_loss,
                },
                global_step=epoch + 1,
            )
            writer.add_scalars(
                main_tag="accuracy",
                tag_scalar_dict={
                    "train": train_accuracy,
                    "val": val_accuracy,
                },
                global_step=epoch + 1,
            )

            progress.set_postfix(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/accuracy": train_accuracy,
                    "val/accuracy": val_accuracy,
                },
                refresh=False,
            )

            # store model
            torch.save(obj=model.state_dict(), f=f"{name}-{(epoch + 1):02d}.pth")

        # compute final evaluation results
        print("After training:")

        bna_train_loss, bna_train_accuracy = test_step(
            model=model,
            data_loader=split2loader["train"],
        )
        test_loss, test_accuracy = test_step(
            model=model,
            data_loader=split2loader["test"],
        )

        showtime(
            model=model,
            data_loader=showtime_loader,
            writer=writer,
            global_step=EPOCHS,
        )

        loss["BA train"].append(bna_train_loss)
        accs["BA train"].append(bna_train_accuracy)
        loss["test"].append(test_loss)
        accs["test"].append(test_accuracy)

        # log to TensorBoard
        writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={
                "BA train": bna_train_loss,
                "test": test_loss,
            },
            global_step=EPOCHS,
        )
        writer.add_scalars(
            main_tag="accuracy",
            tag_scalar_dict={
                "BA train": bna_train_accuracy,
                "test": test_accuracy,
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
                pd.concat(
                    [pd.Series(v).describe() for v in accs.values()],
                    axis=1,
                    keys=[k for k in accs.keys()],
                ),
            ],
            axis=1,
            keys=["loss", "accuracy"],
        )


# %%
# instantiate the network and move it to the chosen device
name: str = "context"
model: ClipContex = ClipContex(
    img_encoder=clip_freezed_img_encoder,
    txt_encoder=clip_freezed_txt_encoder,
).to(device)

# %%
report: pd.DataFrame = training_loop(
    name,
    model,
    lambda params: torch.optim.SGD(params=params, lr=0.01, weight_decay=1e-6, momentum=0.9),
)
report.to_csv(f"training-{name}.csv")


# %%
def compare(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mA[IoU .3]": [
                (report["iou"] >= 0.3).sum() / report["iou"].count()
                for report in reports.values()
            ],
            "mA[IoU .5]": [
                (report["iou"] >= 0.5).sum() / report["iou"].count()
                for report in reports.values()
            ],
            "mA[IoU .7]": [
                (report["iou"] >= 0.7).sum() / report["iou"].count()
                for report in reports.values()
            ],
            "mIoU": [report["iou"].mean() for report in reports.values()],
            "mCos": [report["cos similarity"].mean() for report in reports.values()],
            "mED": [report["euclidean distance"].mean() for report in reports.values()],
        },
        index=reports.keys(),
    )


# %%
eval_reports: dict[Split, pd.DataFrame] = {
    split: eval_step(model, loader, preprocess)
    for split, loader in split2loader.items()
}

# %%
for split, report in eval_reports.items():
    report.describe().to_csv(f"eval-{name}-{split}.csv")

# %%
compare(eval_reports).to_csv("comparing.csv")
