# %% [markdown]
# <a href="https://colab.research.google.com/github/gekoramy/uni.deep-learning/blob/data-augmentation/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
torchinfo
torchvision
tqdm
ultralytics
END

pip install -q -r requirements.txt
pip install -q git+https://github.com/openai/CLIP.git

# %%
import clip
import json
import os
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import PIL
import itertools as it
import math
import matplotlib.pyplot as plt # added in this notebook
import random # added in this notebook

from datetime import datetime
from jaxtyping import Float, UInt, Int
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchinfo import summary
from typing import Literal, Callable, Mapping, TypeVar
from tqdm import tqdm
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

# %%
device: Literal['cpu', 'cuda'] = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
device

# %% [markdown]
# ## Dataset preparation

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
    # segmentation: list[list[float]]  # description of the mask; example [[44.17, 217.83, 36.21, 219.37, 33.64, 214.49, 31.08, 204.74, 36.47, 202.68, 44.17, 203.2]]
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

id2annotation: Mapping[int, Annotation] = {x.id: x for x in instances.annotations}


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

    def __getitem__(
        self, index: int
    ) -> tuple[PIL.Image, list[str], Float[torch.Tensor, "4"]]:
        i, ps, xywh = self.items[index]
        xyxy: Float[torch.Tensor, "4"] = torchvision.ops.box_convert(xywh, in_fmt="xywh", out_fmt="xyxy")
        with PIL.Image.open(i) as img:
            img.load()
            return img, ps, xyxy


# %%
def unzip(batch: list[tuple[T, ...]]) -> tuple[list[T], ...]:
    return tuple(zip(*batch))


# %%
batch_size: int = 3
limit: int = 5 * batch_size

# %%
dl: DataLoader[tuple[list[PIL.Image], list[list[str]], list[Float[torch.Tensor, "4"]]]] = DataLoader(
    dataset=CocoDataset(split="test", limit=limit),
    batch_size=batch_size,
    collate_fn=unzip,
)

# %%
imgs: tuple[PIL.Image, ...]
promptss: tuple[list[str], ...]
true_xyxy: tuple[Float[torch.Tensor, "4"], ...]

for imgs, promptss, true_xyxy in dl:
    print(imgs)
    print(promptss)
    print(true_xyxy)
    print("-" * 50)


# %% [markdown]
# # Image augmentation

# %%
class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        return [random.choice(self.transforms)(img) for img in imgs]


# %%
transform = RandomChoice([
      transforms.ColorJitter(brightness=.5, hue=.3),                # randomly changes the brightness, saturation, and other properties of an image
      transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # performs gaussian blur transform on an image
      transforms.RandomInvert(),                                    # randomly inverts the colors of the given image
      transforms.RandomPosterize(bits=2),                           # randomly posterizes the image by reducing the number of bits of each color channel
      transforms.RandomSolarize(threshold=192.0),                   # randomly solarizes the image by inverting all pixel values above the threshold
      transforms.RandomAdjustSharpness(sharpness_factor=2),         # randomly adjusts the sharpness of the given image
      transforms.RandomAutocontrast(),                              # randomly applies autocontrast to the given image
      transforms.RandomEqualize(),                                  # randomly equalizes the histogram of the given image
      transforms.Grayscale(num_output_channels=3)                   # converts an image to grayscale
    ])

# %%
torch.manual_seed(42)

def display_images(images, titles, rows, cols):
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
  for i, ax in enumerate(axes.flatten()):
      ax.imshow(images[i])
      ax.set_title(titles[i]+str(images[i].size))
      ax.axis('off')
  plt.tight_layout()
  plt.show()

for imgs, promptss, true_xyxy in dl:

    transformed_images = transform(imgs)

    display_images(
      images=list(imgs) + transformed_images,
      titles=['Original Image 1', 'Original Image 2', 'Original Image 3', 'Transformed Image 1', 'Transformed Image 2', 'Transformed Image 3'],
      rows=2,
      cols=3
    )

# %% [markdown]
# # Text augmentation

# %%
# !pip install transformers
# !pip install sentencepiece

# %%
# EDA
# paper: https://aclanthology.org/D19-1670.pdf
# paper: https://arxiv.org/abs/1907.03752
# code reference: https://github.com/dsfsi/textaugment
try:
    from textaugment import EDA
except ModuleNotFoundError:
    # !pip install textaugment
    from textaugment import EDA

import nltk # NLTK is a leading platform for building Python programs to work with human language data
nltk.download('stopwords')
nltk.download('wordnet')

# %%
# PEGASUS fine-tuned for paraphrasing
# paper: https://arxiv.org/abs/1912.08777
# code reference: https://huggingface.co/tuner007/pegasus_paraphrase
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
pegasus_model_name = 'tuner007/pegasus_paraphrase'
pegasus_torch_device = device
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(pegasus_torch_device)


# %%
def pegasus_get_response(input_text,num_return_sequences=1,num_beams=10):
  batch = pegasus_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(pegasus_torch_device)
  translated = pegasus_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


# %%
# A large BART seq2seq (text2text generation) model fine-tuned on 3 paraphrase datasets.
# paper: https://arxiv.org/abs/1910.13461
# code reference: https://huggingface.co/eugenesiow/bart-paraphrase
from transformers import BartForConditionalGeneration, BartTokenizer
bart_model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
bart_torch_device = device
bart_model = bart_model.to(bart_torch_device)
bart_tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')


# %%
class TextAugmentation(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eda_transformation = EDA(random_state=1)  # EDA initialization

    self.transforms = [
        self.eda_synonym_replacement,
        self.template_insertion("A photo of {}"),
        self.template_insertion("A picture of {}"),
        self.template_insertion("An image of {}"),
        self.template_insertion("This is {}"),
        self.template_insertion("We can see {}"),
        self.pegasus_augmentation,
        self.bert_augmentation
    ]

  # randomly choose n words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random
  def eda_synonym_replacement(self, txt):
    return self.eda_transformation.synonym_replacement(txt)

  # add a template as suggested by the CLIP paper. Examples: A photo of {}, We can see {}, ...
  def template_insertion(self, template):
    return template.format

  # pegasus paraphrasing
  def pegasus_augmentation(self, txt):
    return pegasus_get_response(txt)[0]

  # bert paraphrasing
  def bert_augmentation(self, txt):
    batch = bart_tokenizer(txt, return_tensors='pt')
    generated_ids = bart_model.generate(batch['input_ids'])
    generated_sentence = bart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_sentence[0]

  def __call__(self, txts):
      return [random.choice(self.transforms)(txt) for txt in txts]


# %%
torch.manual_seed(42)
text_transform = TextAugmentation()
for imgs, promptss, true_xyxy in dl:
  print("ORIGINAL PROMPTS")
  for prompts in promptss:
    for prompt in prompts:
      print(prompt)

  print("====\n")

  print("TRANSFORMED")
  for prompts in promptss:
    transformed_prompts = text_transform(prompts)
    for prompt in transformed_prompts:
      print(prompt)

  print("\nEND BATCH\n")

# %% [markdown]
# ### ChatGPT experiments

# %%
"""
!pip install openai
import openai

# Set your OpenAI API key here
#openai.api_key = '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'
openai.api_key = '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'

# Define a prompt to start the conversation
prompt = "You are a helpful assistant.\nUser: Hi, can you tell me about the solar system?"

"""
"""
# Generate a response from ChatGPT
response = openai.Completion.create(
    engine="gpt-3.5-turbo",  # Use the appropriate engine for ChatGPT
    prompt=prompt,
    max_tokens=50  # You can adjust this to control the response length
)
"""
"""
response = openai.ChatCompletion.create(
 model="gpt-3.5-turbo",
 messages=[
 {"role": "user", "content": "Hello ChatGPT, does this work?"}
 ]
)
print(response.choices[0].message.content)
"""
"""
# Extract and print the assistant's reply
assistant_reply = response.choices[0].text.strip()
print("Assistant:", assistant_reply)
"""
