# Deep Learning Assignment 2023 - From words to bounding boxes: exploring visual grounding using CLIP
### Academic year 2022-2023
#### Luca Mosetti - luca.mosetti-1@studenti.unitn.it<br>Stefano Genetti - stefano.genetti@studenti.unitn.it

### 📄 REPORT: [report.ipynb](report.ipynb) 

Visual grounding involves linking language and perception by grounding linguistic symbols in the visual world. More in depth, in this work we face the problem usually referred to by the literature as Referring expression comprehension (REC). In this context the overall goal is to localize a target object in an image described by a referring expression phrased in natural language. In order to accomplish this challenging task we rely on the [CLIP (Contrastive Language-Image Pre-training)](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Farxiv.org%2Fabs%2F2103.00020) pre-trained model as a starting point for transfer learning. The capabilities of this foundation model pose a starting point to design a joint embedding approach to solve the problem at hand. In this work we provide an overview of the strategies which we have adopted in order to fine-tune CLIP for the task under discussion.
More in depth, three distinct architectures have been proposed: a conventional fine-tuning approach, a contrastive learning method inspired by the "fine-tune like you pretrain" concept, and a self-attention-based approach.
We have evaluated our proposed models on the commonly used [RefCOCOg dataset](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Farxiv.org%2Fabs%2F1608.00272). In addition to this, our contribution is to provide three useful instances of the dataset filled with the bounding boxes proposed by some well known object detection algorithms. As further explained in the (report)[report.ipynb] this solution allows to considerably speed up the training procedure. We conveniently provide these datasets together with the code to generate them at [this GitHub repository](https://github.com/StefanoGenettiUniTN/refcocog-augmentation).<br>

**The overall work is exhaustively detailed in [report.ipynb](report.ipynb) in which we alternate the text cells with code cells incorporating the implemented code.**

![approach](approach.png)
