# Read the COCO dataset
In this branch we propose the code to read the **RefCOCOg dataset**, a variant of the Referring Expression Generation (REG) dataset.



# File structure :page_with_curl:
- **images**:
  - images are in `refcocog/images`
  - each image file is named:
    - `COCO_train2014_[image_id]_[annotation_id].jpg`
    - Example: `COCO_train2014_000000130518_104426.jpg`
  - according to the documentation online, the dataset is split as follows:
    - test set 2600 images
    - training set 21899 images
    - validation set 1300 images
    - (read `refs(umd).p` to understand whether an image belongs to the test, train or validation set)
- **instances.json**: the file is in `refcocog/annotations`. The file is hierarchically structured as follows:
  ```
  - info
    - description: This is stable 1.0 version of the 2014 MS COCO dataset.
    - url: http://mscoco.org/
    - version: 1.0
    - year: 2014
    - contributor: Microsoft COCO group
    - data_created: 2015-01-27 09:11:52.357475
  - images (list of images, the length of the list is 25799=train 21899 + val 1300 + test 2600)
    - license: each image has an associated licence id
    - file_name: file name of the image
    - coco_url: example http://mscoco.org/images/131074
    - height
    - width
    - data_captured: example '2013-11-21 01:03:06'
    - flickr_url: example http://farm9.staticflickr.com/8308/7908210548_33e
    - id: id of the image
  - licenses
    - url: example http://creativecommons.org/licenses/by-nc-sa/2.0/
    - id: id of the licence
    - name: example 'Attribution-NonCommercial-ShareAlike License'
  - annotations (list of image annotations, the length of the list is 208960)
    - segmentation: description of the mask; example [[44.17, 217.83, 36.21, 219.37, 33.64, 214.49, 31.08, 204.74, 36.47, 202.68, 44.17, 203.2]]
    - area: number of pixel of the described object
    - iscrowd: value is 1 or 0; Crowd annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people)
    - image_id: id of the target image
    - bbox: bounding box coordinates [xmin, ymin, width, height]
    - category_id
    - id: annotation id
  - categories (list of categories)
    - supercategory: example 'vehicle'
    - id: category id
    - name: example 'airplane'
  ```
- **refs(umd).p** the file is in `refcocog/annotations`. In total the file contains 49822 objects. The file is hierarchically structured as follows:
  ```
  - image_id: image id
  - split: train, test or val
  - sentences: list of sentence objects
    - tokens: ['a', 'man', 'with', 'black', 'glasses', 'sitting', 'in', 'between', 'two', 'other', 'men']
    - raw: A man with black glasses sitting in between two other men. (unprocessed referring expression (str))
    - sent_id: 20049 (sentence id)
    - sent: a man with black glasses sitting in between two other men (referring expression with mild processing, lower case, spell correction, etc. (str))
  - file_name: target image file
  - category_id
  - ann_id: annotation id
  - sent_ids: [20049, 20050] (same ids as nested sentences[...][sent_id] (list of int))
  - ref_id: unique id for this refering expression (int)
  ```
Online I found this description of the pickle file:
```
refs: list of dict [
    {
    image_id : unique image id (int)
    split : train/test/val (str)
    sentences : list of dict [
        {
        tokens : tokenized version of referring expression (list of str)
        raw : unprocessed referring expression (str)
        sent : referring expression with mild processing, lower case, spell correction, etc. (str)
        sent_id : unique referring expression id (int)
        } ...
    ]
    file_name : file name of image relative to img_root (str)
    category_id : object category label (int)
    ann_id : id of object annotation in instance.json (int)
    sent_ids : same ids as nested sentences[...][sent_id] (list of int)
    ref_id : unique id for refering expression (int)
    } ...
] 
```
