import enum
import os
import glob
import json
import copy
import textwrap
import logging
import zipfile
import functools
import re
from PIL import Image
import datasets

logger = logging.getLogger(__name__)

_VERSION = datasets.Version("1.0.0", "")

_URL = "https://visualqa.org/"

_CITATION = """\
```
@InProceedings{balanced_vqa_v2,
author = {Yash Goyal and Tejas Khot and Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},
title = {Making the {V} in {VQA} Matter: Elevating the Role of Image Understanding in {V}isual {Q}uestion {A}nswering},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017},
}
```

```
@InProceedings{balanced_binary_vqa,
author = {Peng Zhang and Yash Goyal and Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},
title = {{Y}in and {Y}ang: Balancing and Answering Binary Visual Questions},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2016},
}
```

```
@InProceedings{{VQA},
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {{VQA}: {V}isual {Q}uestion {A}nswering},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2015},
}
```
"""

_VQA_V2_CITATION = """
```
@InProceedings{balanced_vqa_v2,
author = {Yash Goyal and Tejas Khot and Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},
title = {Making the {V} in {VQA} Matter: Elevating the Role of Image Understanding in {V}isual {Q}uestion {A}nswering},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017},
}
```
"""

_VQA_V1_CITATION = """
```
@InProceedings{{VQA},
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {{VQA}: {V}isual {Q}uestion {A}nswering},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2015},
}
```
"""

_VQA_BALANCED_BIN_ABST_CITATION = """
```
@InProceedings{balanced_binary_vqa,
author = {Peng Zhang and Yash Goyal and Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},
title = {{Y}in and {Y}ang: Balancing and Answering Binary Visual Questions},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2016},
}
```
"""

_DESCRIPTION = """\
# VQA

## What is VQA?
VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
- 265,016 images (COCO and abstract scenes)
- At least 3 questions (5.4 questions on average) per image
- 10 ground truth answers per question
- 3 plausible (but likely incorrect) answers per question
- Automatic evaluation metric

## Dataset
Details on downloading the latest dataset may be found on the [download webpage](https://visualqa.org/download.html).

## Usage
```python
from datasets import load_dataset

raw_datasets = load_dataset(
                "vqa.py", 
                "base",
                cache_dir="huggingface_datasets", 
                data_dir="data",
                ignore_verifications=True,
            )

dataset_train = raw_datasets["train"]

for item in dataset_train:
    print(item)
    exit()
```

v2 = v2.real + v2.abstract (v2.abstract == v1.abstract)
v1 = v1.real + v1.abstract
v2.abstract.balanced.bin
"""

# training data path
BALANCED_REAL_ANNO_V2_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
BALANCED_REAL_ANNO_V2_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
BALANCED_REAL_Q_V2_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
BALANCED_REAL_Q_V2_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
BALANCED_REAL_Q_V2_TEST_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
REAL_IMGS_TRAINING_URL = "http://images.cocodataset.org/zips/train2014.zip"
REAL_IMGS_VALIDATION_URL = "http://images.cocodataset.org/zips/val2014.zip"
REAL_IMGS_TEST_URL = "http://images.cocodataset.org/zips/test2015.zip"
BALANCED_REAL_COMP_PAIRS_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip"
BALANCED_REAL_COMP_PAIRS_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip"

BALANCED_BIN_ABST_ANNO_V2_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Binary_Train2017_abstract_v002.zip"
BALANCED_BIN_ABST_ANNO_V2_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Binary_Val2017_abstract_v002.zip"
BALANCED_BIN_ABST_Q_V2_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Binary_Train2017_abstract_v002.zip"
BALANCED_BIN_ABST_Q_V2_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Binary_Val2017_abstract_v002.zip"
BALANCED_BIN_ABST_IMGS_V2_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_binary_train2017.zip"
BALANCED_BIN_ABST_IMGS_V2_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_binary_val2017.zip"

# abstract scenes (same as v1.0 release)
ABST_ANNO_V1_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Train_abstract_v002.zip"
ABST_ANNO_V1_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip"
ABST_Q_V1_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Train_abstract_v002.zip"
ABST_Q_V1_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip"
ABST_Q_V1_TEST_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Test_abstract_v002.zip"
ABST_IMGS_V1_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip"
ABST_IMGS_V1_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip"
ABST_IMGS_V1_TEST_URL = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_test2015.zip"

# real images for v1.0
REAL_ANNO_V1_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip"
REAL_ANNO_V1_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip"
REAL_Q_V1_TRAINING_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip"
REAL_Q_V1_VALIDATION_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip"
REAL_Q_V1_TEST_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip"


# *** file name ***
BALANCED_REAL_ANNO_V2_TRAINING_FNAME = "v2_mscoco_train2014_annotations.json"
BALANCED_REAL_ANNO_V2_VALIDATION_FNAME = "v2_mscoco_val2014_annotations.json"
BALANCED_REAL_Q_V2_TRAINING_FNAME = "v2_OpenEnded_mscoco_train2014_questions.json"
BALANCED_REAL_Q_V2_VALIDATION_FNAME = "v2_OpenEnded_mscoco_val2014_questions.json"
BALANCED_REAL_Q_V2_TEST_FNAME = "v2_OpenEnded_mscoco_test2015_questions.json"
BALANCED_REAL_Q_V2_TEST_DEV_FNAME = "v2_OpenEnded_mscoco_test-dev2015_questions.json"
REAL_IMGS_TRAINING_FNAME = "train2014"
REAL_IMGS_VALIDATION_FNAME = "val2014"
REAL_IMGS_TEST_FNAME = "test2015"
BALANCED_REAL_COMP_PAIRS_TRAINING_FNAME = "v2_mscoco_train2014_complementary_pairs.json"
BALANCED_REAL_COMP_PAIRS_VALIDATION_FNAME = "v2_mscoco_val2014_complementary_pairs.json"

BALANCED_BIN_ABST_ANNO_V2_TRAINING_FNAME = "abstract_v002_train2017_annotations.json"
BALANCED_BIN_ABST_ANNO_V2_VALIDATION_FNAME = "abstract_v002_val2017_annotations.json"
BALANCED_BIN_ABST_Q_V2_TRAINING_FNAME = "OpenEnded_abstract_v002_train2017_questions.json"
BALANCED_BIN_ABST_Q_V2_VALIDATION_FNAME = "OpenEnded_abstract_v002_val2017_questions.json"
BALANCED_BIN_ABST_IMGS_V2_TRAINING_FNAME = "scene_img_abstract_v002_train2017"
BALANCED_BIN_ABST_IMGS_V2_VALIDATION_FNAME = "scene_img_abstract_v002_val2017"

# abstract scenes (same as v1.0 release)
ABST_ANNO_V1_TRAINING_FNAME = "abstract_v002_train2015_annotations.json"
ABST_ANNO_V1_VALIDATION_FNAME = "abstract_v002_val2015_annotations.json"
ABST_Q_V1_TRAINING_FNAME = "OpenEnded_abstract_v002_train2015_questions.json"
ABST_Q_V1_VALIDATION_FNAME = "OpenEnded_abstract_v002_val2015_questions.json"
ABST_Q_V1_TEST_FNAME = "OpenEnded_abstract_v002_test2015_questions.json"
ABST_IMGS_V1_TRAINING_FNAME = "./"
ABST_IMGS_V1_VALIDATION_FNAME = "./"
ABST_IMGS_V1_TEST_FNAME = "./"

# real images for v1.0
REAL_ANNO_V1_TRAINING_FNAME = "mscoco_train2014_annotations.json"
REAL_ANNO_V1_VALIDATION_FNAME = "mscoco_val2014_annotations.json"
REAL_Q_V1_TRAINING_FNAME = "OpenEnded_mscoco_train2014_questions.json"
REAL_Q_V1_VALIDATION_FNAME = "OpenEnded_mscoco_val2014_questions.json"
REAL_Q_V1_TEST_FNAME = "OpenEnded_mscoco_test2015_questions.json"
REAL_Q_V1_TEST_DEV_FNAME = "OpenEnded_mscoco_test-dev2015_questions.json"

# multiple choice
REAL_Q_V1_MC_TRAINING_FNAME = "MultipleChoice_mscoco_train2014_questions.json"
REAL_Q_V1_MC_VALIDATION_FNAME = "MultipleChoice_mscoco_val2014_questions.json"
REAL_Q_V1_MC_TEST_FNAME = "MultipleChoice_mscoco_test2015_questions.json"
REAL_Q_V1_MC_TEST_DEV_FNAME = "MultipleChoice_mscoco_test-dev2015_questions.json"

ABST_Q_V1_MC_TRAINING_FNAME = "MultipleChoice_abstract_v002_train2015_questions.json"
ABST_Q_V1_MC_VALIDATION_FNAME = "MultipleChoice_abstract_v002_val2015_questions.json"
ABST_Q_V1_MC_TEST_FNAME = "MultipleChoice_abstract_v002_test2015_questions.json"


BALANCED_REAL_V2_URLS = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_URL],
        "question_url": [BALANCED_REAL_Q_V2_TRAINING_URL],
        "annotation_url": [BALANCED_REAL_ANNO_V2_TRAINING_URL],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_URL],
        "question_url": [BALANCED_REAL_Q_V2_VALIDATION_URL],
        "annotation_url": [BALANCED_REAL_ANNO_V2_VALIDATION_URL],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_URL],
        "question_url": [BALANCED_REAL_Q_V2_TEST_URL],
    },
}
BALANCED_REAL_V2_FILE_MAP = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_TRAINING_FNAME],
        "annotation_url": [BALANCED_REAL_ANNO_V2_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_VALIDATION_FNAME],
        "annotation_url": [BALANCED_REAL_ANNO_V2_VALIDATION_FNAME],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_TEST_FNAME],
    },
}

BALANCED_BIN_ABST_V2_URLS = {
    "train": {
        "image_url": [BALANCED_BIN_ABST_IMGS_V2_TRAINING_URL],
        "question_url": [BALANCED_BIN_ABST_Q_V2_TRAINING_URL],
        "annotation_url": [BALANCED_BIN_ABST_ANNO_V2_TRAINING_URL],
    },
    "validation": {
        "image_url": [BALANCED_BIN_ABST_IMGS_V2_VALIDATION_URL],
        "question_url": [BALANCED_BIN_ABST_Q_V2_VALIDATION_URL],
        "annotation_url": [BALANCED_BIN_ABST_ANNO_V2_VALIDATION_URL],
    },
}
BALANCED_BIN_ABST_V2_FILE_MAP = {
    "train": {
        "image_url": [BALANCED_BIN_ABST_IMGS_V2_TRAINING_FNAME],
        "question_url": [BALANCED_BIN_ABST_Q_V2_TRAINING_FNAME],
        "annotation_url": [BALANCED_BIN_ABST_ANNO_V2_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [BALANCED_BIN_ABST_IMGS_V2_VALIDATION_FNAME],
        "question_url": [BALANCED_BIN_ABST_Q_V2_VALIDATION_FNAME],
        "annotation_url": [BALANCED_BIN_ABST_ANNO_V2_VALIDATION_FNAME],
    },
}

ABST_V1V2_URLS = {
    "train": {
        "image_url": [ABST_IMGS_V1_TRAINING_URL],
        "question_url": [ABST_Q_V1_TRAINING_URL],
        "annotation_url": [ABST_ANNO_V1_TRAINING_URL],
    },
    "validation": {
        "image_url": [ABST_IMGS_V1_VALIDATION_URL],
        "question_url": [ABST_Q_V1_VALIDATION_URL],
        "annotation_url": [ABST_ANNO_V1_VALIDATION_URL],
    },
    "test": {
        "image_url": [ABST_IMGS_V1_TEST_URL],
        "question_url": [ABST_Q_V1_TEST_URL],
    },
}
ABST_V1V2_FILE_MAP = {
    "train": {
        "image_url": [ABST_IMGS_V1_TRAINING_FNAME],
        "question_url": [ABST_Q_V1_TRAINING_FNAME],
        "annotation_url": [ABST_ANNO_V1_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [ABST_IMGS_V1_VALIDATION_FNAME],
        "question_url": [ABST_Q_V1_VALIDATION_FNAME],
        "annotation_url": [ABST_ANNO_V1_VALIDATION_FNAME],
    },
    "test": {
        "image_url": [ABST_IMGS_V1_TEST_FNAME],
        "question_url": [ABST_Q_V1_TEST_FNAME],
    },
}

REAL_V1_URLS = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_URL],
        "question_url": [REAL_Q_V1_TRAINING_URL],
        "annotation_url": [REAL_ANNO_V1_TRAINING_URL],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_URL],
        "question_url": [REAL_Q_V1_VALIDATION_URL],
        "annotation_url": [REAL_ANNO_V1_VALIDATION_URL],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_URL],
        "question_url": [REAL_Q_V1_TEST_URL],
    },
}
REAL_V1_FILE_MAP = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_FNAME],
        "question_url": [REAL_Q_V1_TRAINING_FNAME],
        "annotation_url": [REAL_ANNO_V1_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_FNAME],
        "question_url": [REAL_Q_V1_VALIDATION_FNAME],
        "annotation_url": [REAL_ANNO_V1_VALIDATION_FNAME],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_FNAME],
        "question_url": [REAL_Q_V1_TEST_FNAME],
    },
}

V2_URLS = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_URL, ABST_IMGS_V1_TRAINING_URL],
        "question_url": [BALANCED_REAL_Q_V2_TRAINING_URL, ABST_Q_V1_TRAINING_URL],
        "annotation_url": [BALANCED_REAL_ANNO_V2_TRAINING_URL, ABST_ANNO_V1_TRAINING_URL],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_URL, ABST_IMGS_V1_VALIDATION_URL],
        "question_url": [BALANCED_REAL_Q_V2_VALIDATION_URL, ABST_Q_V1_VALIDATION_URL],
        "annotation_url": [BALANCED_REAL_ANNO_V2_VALIDATION_URL, ABST_ANNO_V1_VALIDATION_URL],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_URL, ABST_IMGS_V1_TEST_URL],
        "question_url": [BALANCED_REAL_Q_V2_TEST_URL, ABST_Q_V1_TEST_URL],
    },
}
V2_FILE_MAP = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_FNAME, ABST_IMGS_V1_TRAINING_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_TRAINING_FNAME, ABST_Q_V1_TRAINING_FNAME],
        "annotation_url": [BALANCED_REAL_ANNO_V2_TRAINING_FNAME, ABST_ANNO_V1_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_FNAME, ABST_IMGS_V1_VALIDATION_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_VALIDATION_FNAME, ABST_Q_V1_VALIDATION_FNAME],
        "annotation_url": [BALANCED_REAL_ANNO_V2_VALIDATION_FNAME, ABST_ANNO_V1_VALIDATION_FNAME],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_FNAME, ABST_IMGS_V1_TEST_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_TEST_FNAME, ABST_Q_V1_TEST_FNAME],
    },
}

V1_URLS = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_URL, ABST_IMGS_V1_TRAINING_URL],
        "question_url": [REAL_Q_V1_TRAINING_URL, ABST_Q_V1_TRAINING_URL],
        "annotation_url": [REAL_ANNO_V1_TRAINING_URL, ABST_ANNO_V1_TRAINING_URL],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_URL, ABST_IMGS_V1_VALIDATION_URL],
        "question_url": [REAL_Q_V1_VALIDATION_URL, ABST_Q_V1_VALIDATION_URL],
        "annotation_url": [REAL_ANNO_V1_VALIDATION_URL, ABST_ANNO_V1_VALIDATION_URL],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_URL, ABST_IMGS_V1_TEST_URL],
        "question_url": [REAL_Q_V1_TEST_URL, ABST_Q_V1_TEST_URL],
    },
}
V1_FILE_MAP = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_FNAME, ABST_IMGS_V1_TRAINING_FNAME],
        "question_url": [REAL_Q_V1_TRAINING_FNAME, ABST_Q_V1_TRAINING_FNAME],
        "annotation_url": [REAL_ANNO_V1_TRAINING_FNAME, ABST_ANNO_V1_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_FNAME, ABST_IMGS_V1_VALIDATION_FNAME],
        "question_url": [REAL_Q_V1_VALIDATION_FNAME, ABST_Q_V1_VALIDATION_FNAME],
        "annotation_url": [REAL_ANNO_V1_VALIDATION_FNAME, ABST_ANNO_V1_VALIDATION_FNAME],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_FNAME, ABST_IMGS_V1_TEST_FNAME],
        "question_url": [REAL_Q_V1_TEST_FNAME, ABST_Q_V1_TEST_FNAME],
    },
}
V1_MC_FILE_MAP = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_FNAME, ABST_IMGS_V1_TRAINING_FNAME],
        "question_url": [REAL_Q_V1_MC_TRAINING_FNAME, ABST_Q_V1_MC_TRAINING_FNAME],
        "annotation_url": [REAL_ANNO_V1_TRAINING_FNAME, ABST_ANNO_V1_TRAINING_FNAME],
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_FNAME, ABST_IMGS_V1_VALIDATION_FNAME],
        "question_url": [REAL_Q_V1_MC_VALIDATION_FNAME, ABST_Q_V1_MC_VALIDATION_FNAME],
        "annotation_url": [REAL_ANNO_V1_VALIDATION_FNAME, ABST_ANNO_V1_VALIDATION_FNAME],
    },
    "test": {
        "image_url": [REAL_IMGS_TEST_FNAME, ABST_IMGS_V1_TEST_FNAME],
        "question_url": [REAL_Q_V1_MC_TEST_FNAME, ABST_Q_V1_MC_TEST_FNAME],
    },
}

BALANCED_REAL_COMP_PAIRS_URLS = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_URL, ABST_IMGS_V1_TRAINING_URL],
        "question_url": [BALANCED_REAL_Q_V2_TRAINING_URL, ABST_Q_V1_TRAINING_URL],
        "annotation_url": [BALANCED_REAL_ANNO_V2_TRAINING_URL, ABST_ANNO_V1_TRAINING_URL],
        "pair_url": [BALANCED_REAL_COMP_PAIRS_TRAINING_URL]
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_URL, ABST_IMGS_V1_VALIDATION_URL],
        "question_url": [BALANCED_REAL_Q_V2_VALIDATION_URL, ABST_Q_V1_VALIDATION_URL],
        "annotation_url": [BALANCED_REAL_ANNO_V2_VALIDATION_URL, ABST_ANNO_V1_VALIDATION_URL],
        "pair_url": [BALANCED_REAL_COMP_PAIRS_VALIDATION_URL]
    },
}
BALANCED_REAL_COMP_PAIRS_FILE_MAP = {
    "train": {
        "image_url": [REAL_IMGS_TRAINING_FNAME, ABST_IMGS_V1_TRAINING_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_TRAINING_FNAME, ABST_Q_V1_TRAINING_FNAME],
        "annotation_url": [BALANCED_REAL_ANNO_V2_TRAINING_FNAME, ABST_ANNO_V1_TRAINING_FNAME],
        "pair_url": [BALANCED_REAL_COMP_PAIRS_TRAINING_FNAME]
    },
    "validation": {
        "image_url": [REAL_IMGS_VALIDATION_FNAME, ABST_IMGS_V1_VALIDATION_FNAME],
        "question_url": [BALANCED_REAL_Q_V2_VALIDATION_FNAME, ABST_Q_V1_VALIDATION_FNAME],
        "annotation_url": [BALANCED_REAL_ANNO_V2_VALIDATION_FNAME, ABST_ANNO_V1_VALIDATION_FNAME],
        "pair_url": [BALANCED_REAL_COMP_PAIRS_VALIDATION_FNAME]
    },
}


# License: Creative Commons Attribution 4.0 International License

def create_img_kv(dir_path):
    img_kv = {}
    for type_wildcard in ["*.png", "*.jpg", "*.jpeg"]:
        for fname in glob.glob(os.path.join(dir_path, type_wildcard)):
            img_name, _ = os.path.splitext(os.path.basename(fname))
            img_id = int(img_name.split("_")[-1])
            img_kv[img_id] = fname
    return img_kv

def parsing_common_info(item):
    _info = item["info"]
    _data_type = item["data_type"]
    _data_subtype = item["data_subtype"]
    _license = item["license"]

    return {
        "info": _info,
        "data_type": _data_type,
        "data_subtype": _data_subtype,
        "license": _license,
    }

def parsing_questions(fname, is_mc=False):
    data = json.load(open(fname, "r"))
    common_info = parsing_common_info(data)
    _questions = data["questions"]
    for q in _questions:
        item =  {
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "question": q["question"],
            "data_type": common_info["data_type"],
            "data_subtype": common_info["data_subtype"],
        }
        if is_mc:
            item["multiple_choices"] = q["multiple_choices"]
        yield item

def parsing_annotations(fname):
    if fname is None:
        return None
    anno_info = {}
    item = json.load(open(fname, "r"))
    _annotations = item["annotations"]
    for _anno in _annotations:
        anno_info[_anno["question_id"]] = _anno
    return anno_info

def verifying_answer_format(answers):
    for i in range(len(answers)):
        for k, v in answers[i].items():
            answers[i][str(k)] = re.sub(r'[^a-zA-Z0-9 ]', '', str(v))

    if "answer_confidence" in answers[0]:
        return answers
    else:
        for idx in range(len(answers)):
            answers[idx]["answer_confidence"] = "yes"
        return answers


def parse_samples(extracted_files, is_mc=False):
    question_files = extracted_files["question_url"]
    image_dirs = extracted_files["image_url"]
    if "annotation_url" in extracted_files:
        annotation_files = extracted_files["annotation_url"]
    else:
        annotation_files = [None] * len(question_files)

    for question_file, annotation_file, image_idr in zip(question_files, annotation_files, image_dirs):
        annos = parsing_annotations(annotation_file)
        img_kv = create_img_kv(image_idr)
        for item in parsing_questions(question_file, is_mc=is_mc):
            question_id = item["question_id"]
            image_id = item["image_id"]

            image_path = img_kv.get(image_id)

            if annos is not None:
                anno = annos.get(question_id)
            else:
                anno = None

            parsed_sample = {
                "image_id": image_id,
                "question_id": item["question_id"],
                "question": item["question"],
                "question_type": anno["question_type"] if anno is not None else None,
                "answers": verifying_answer_format(anno["answers"]) if anno is not None else None,
                "answer_type": anno["answer_type"] if anno is not None else None,
                "multiple_choice_answer": re.sub(r'[^a-zA-Z0-9 ]', '', anno["multiple_choice_answer"]) if anno is not None else None,
            }
            if is_mc:
                parsed_sample["multiple_choices"] = item["multiple_choices"]
            yield image_path, parsed_sample

def parse_samples_dd(extracted_files, is_mc=False):
    question_files = extracted_files["question_url"]
    image_dirs = extracted_files["image_url"]
    if "annotation_url" in extracted_files:
        annotation_files = extracted_files["annotation_url"]
    else:
        annotation_files = [None] * len(question_files)

    for question_file, annotation_file, image_idr in zip(question_files, annotation_files, image_dirs):
        annos = parsing_annotations(annotation_file)
        img_kv = create_img_kv(image_idr)
        for item in parsing_questions(question_file, is_mc=is_mc):
            question_id = item["question_id"]
            image_id = item["image_id"]

            image_path = img_kv.get(image_id)

            if annos is not None:
                anno = annos.get(question_id)
            else:
                anno = None

            parsed_sample = {
                "image_id": image_id,
                "question_id": item["question_id"],
                "question": item["question"],
                "question_type": anno["question_type"] if anno is not None else None,
                "answers": verifying_answer_format(anno["answers"]) if anno is not None else None,
                "answer_type": anno["answer_type"] if anno is not None else None,
                "multiple_choice_answer": anno["multiple_choice_answer"] if anno is not None else None,
            }
            if is_mc:
                parsed_sample["multiple_choices"] = item["multiple_choices"]
            yield image_path, parsed_sample

def generator(extracted_files, is_mc=False, convert2rgb=False):
    for image_path, item in parse_samples(extracted_files, is_mc=is_mc):
        if convert2rgb:
            item["image"] = Image.open(image_path).convert("RGB")
        else:
            item["image"] = {
                "path": image_path,
                "bytes": open(image_path, "rb").read(),
            }
        yield item

def generator_for_comp_pairs(extracted_files):
    pair_url = extracted_files["pair_url"]
    q_id_pairs = json.load(open(pair_url[0]))

    item_kv = {}
    for image_path, item in parse_samples(extracted_files, is_mc=False):
        item["image"] = image_path
        item_kv[item["question_id"]] = item
    
    for qid1, qid2 in q_id_pairs:
        sample1 = copy.deepcopy(item_kv.get(qid1))
        sample2 = copy.deepcopy(item_kv.get(qid2))

        image_path1 = sample1["image"]
        sample1["image"] = {
            "path": image_path1,
            "bytes": open(image_path1, "rb").read(),
        }
        image_path2 = sample2["image"]
        sample2["image"] = {
            "path": image_path2,
            "bytes": open(image_path2, "rb").read(),
        }

        yield {
            "sample1": sample1,
            "sample2": sample2,
        }


def get_dicts_for_label(extracted_files, is_mc=False, convert2rgb=False):
    question_files = extracted_files["question_url"]
    image_dirs = extracted_files["image_url"]

    if "annotation_url" in extracted_files:
        annotation_files = extracted_files["annotation_url"]
    else:
        annotation_files = [None] * len(question_files)
    answer_list = []
    answer_confidence_list = []
    multiple_choice_list = []
    for question_file, annotation_file, image_idr in zip(question_files, annotation_files, image_dirs):
        annos = parsing_annotations(annotation_file)
        for idx, item in enumerate(parsing_questions(question_file, is_mc=is_mc)):
            question_id = item["question_id"]
            if annos is not None:
                anno = annos.get(question_id)
            else:
                anno = None
            # anno: {
            #     'question_type': 'what',
            #     'multiple_choice_answer': 'curved',
            #     'answers': [
            #         {
            #             'answer': 'oval',
            #             'answer_confidence': 'yes',
            #             'answer_id': 1
            #         }, {
            #             'answer': 'semi circle',
            #             'answer_confidence': 'yes',
            #             'answer_id': 2
            #         }, ... , {
            #             'answer': 'curved',
            #             'answer_confidence': 'maybe',
            #             'answer_id': 10
            #         }
            #     ],
            #     'image_id': 487025,
            #     'answer_type': 'other',
            #     'question_id': 4870250
            # }
            answers = verifying_answer_format(anno["answers"])
            for i in range(len(answers)):
                answer_list.append(answers[i]["answer"])
                answer_confidence_list.append(answers[i]["answer_confidence"])
            
            multiple_choice_list.append(re.sub(r'[^a-zA-Z0-9 ]', '', anno["multiple_choice_answer"]))
            if idx % 100 == 0:
                answer_list = list(dict.fromkeys(answer_list))
                answer_confidence_list = list(dict.fromkeys(answer_confidence_list))
                multiple_choice_list = list(dict.fromkeys(multiple_choice_list))
    answer_list = sorted(list(dict.fromkeys(answer_list)))
    answer_confidence_list = sorted(list(dict.fromkeys(answer_confidence_list)))
    multiple_choice_list = sorted(list(dict.fromkeys(multiple_choice_list)))
    return answer_list, answer_confidence_list, multiple_choice_list

# question_type, answer_type
DEFAULT_FEATURES=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_type": datasets.Value("string"),
                    "answers": datasets.Sequence({
                        "answer_id": datasets.Value("int32"),
                        "answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                    }),
                    "answer_type": datasets.Value("string"),
                    "multiple_choice_answer": datasets.Value("string"),
                }
            )

MC_FEATURES=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_type": datasets.Value("string"),
                    "answers": datasets.Sequence({
                        "answer_id": datasets.Value("int32"),
                        "answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                    }),
                    "answer_type": datasets.Value("string"),
                    "multiple_choice_answer": datasets.Value("string"),
                    "multiple_choices": datasets.Sequence(datasets.Value("string")),
                }
            )

with open("keti_air_vqa/answer_list.json", "r") as DE_DUPLICATED_ANSWER_CAT_JSON:
    DE_DUPLICATED_ANSWER_CAT = json.load(DE_DUPLICATED_ANSWER_CAT_JSON)
# guesswhat cat info
with open("keti_air_vqa/answer_confidence_list.json", "r") as DE_DUPLICATED_ANSWER_CONFIDENCE_CAT_JSON:
    DE_DUPLICATED_ANSWER_CONFIDENCE_CAT = json.load(DE_DUPLICATED_ANSWER_CONFIDENCE_CAT_JSON)

with open("keti_air_vqa/multiple_choice_list.json", "r") as DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER_CAT_JSON:
    DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER_CAT = json.load(DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER_CAT_JSON)

print("[*] class_length: ", len(DE_DUPLICATED_ANSWER_CAT), len(DE_DUPLICATED_ANSWER_CONFIDENCE_CAT), len(DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER_CAT))

DE_DUPLICATED_ANSWER = datasets.ClassLabel(
    num_classes=len(DE_DUPLICATED_ANSWER_CAT),
    names=DE_DUPLICATED_ANSWER_CAT
)
DE_DUPLICATED_ANSWER_CONFIDENCE = datasets.ClassLabel(
    num_classes=len(DE_DUPLICATED_ANSWER_CONFIDENCE_CAT),
    names=DE_DUPLICATED_ANSWER_CONFIDENCE_CAT
)
DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER = datasets.ClassLabel(
      num_classes=len(DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER_CAT),
    names=DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER_CAT  
)

DEFAULT_FEATURES_DD=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_type": datasets.Value("string"),
                    "answers": datasets.Sequence({
                        "answer_id": datasets.Value("int32"),
                        "answer": DE_DUPLICATED_ANSWER,
                        "answer_confidence": DE_DUPLICATED_ANSWER_CONFIDENCE,
                    }),
                    "answer_type": datasets.Value("string"),
                    "multiple_choice_answer": DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER,
                }
            )

DE_DUPLICATED_MC_FEATURES=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_type": datasets.Value("string"),
                    "answers": datasets.Sequence({
                        "answer_id": datasets.Value("int32"),
                        "answer": DE_DUPLICATED_ANSWER,
                        "answer_confidence": DE_DUPLICATED_ANSWER_CONFIDENCE,
                    }),
                    "answer_type": datasets.Value("string"),
                    "multiple_choice_answer": DE_DUPLICATED_MULTIPLE_CHOICE_ANSWER,
                    "multiple_choices": datasets.Sequence(datasets.Value("string")),
                }
            )

# complementary.pairs
COMP_PAIRS_FEATURES=datasets.Features(
                {
                    "sample1": DEFAULT_FEATURES,
                    "sample2": DEFAULT_FEATURES,
                }
            )

COMP_PAIRS_FEATURES_DD=datasets.Features(
                {
                    "sample1": DEFAULT_FEATURES_DD,
                    "sample2": DEFAULT_FEATURES_DD,
                }
            )

class VQAConfig(datasets.BuilderConfig):
    """BuilderConfig for VQA."""

    def __init__(
            self, 
            data_urls=V2_URLS,
            file_map=V2_FILE_MAP,
            citation=_VQA_V2_CITATION,
            features=DEFAULT_FEATURES,
            **kwargs):
        """BuilderConfig for VisualInfoVQA.

        Args:
        features: datasets.Feature for the dataset.
        **kwargs: keyword arguments forwarded to super.
        """
        super(VQAConfig, self).__init__(**kwargs)
        self.data_urls = data_urls
        self.file_map = file_map
        self.citation = citation
        self.features = features


class VQA(datasets.GeneratorBasedBuilder):
    """VQA Dataset"""

    BUILDER_CONFIGS = [
        VQAConfig(
            data_urls=V2_URLS,
            file_map=V2_FILE_MAP,
            citation=_VQA_V2_CITATION,
            features=DEFAULT_FEATURES_DD,
            name="v2.dd",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=V1_URLS,
            file_map=V1_FILE_MAP,
            citation=_VQA_V1_CITATION,
            features=DEFAULT_FEATURES_DD,
            name="v1.dd",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=BALANCED_BIN_ABST_V2_URLS,
            file_map=BALANCED_BIN_ABST_V2_FILE_MAP,
            citation=_VQA_BALANCED_BIN_ABST_CITATION,
            features=DEFAULT_FEATURES_DD,
            name="balanced.binary.abstract.dd",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=BALANCED_REAL_COMP_PAIRS_URLS,
            file_map=BALANCED_REAL_COMP_PAIRS_FILE_MAP,
            citation=_VQA_V2_CITATION,
            features=COMP_PAIRS_FEATURES,
            name="complementary.pairs.dd",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=V1_URLS,
            file_map=V1_MC_FILE_MAP,
            citation=_VQA_V1_CITATION,
            features=DE_DUPLICATED_MC_FEATURES,
            name="v1.mc.dd",
            version=_VERSION,
        ),                                        
        VQAConfig(
            data_urls=V2_URLS,
            file_map=V2_FILE_MAP,
            citation=_VQA_V2_CITATION,
            features=DEFAULT_FEATURES,
            name="v2",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=V1_URLS,
            file_map=V1_FILE_MAP,
            citation=_VQA_V1_CITATION,
            features=DEFAULT_FEATURES,
            name="v1",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=BALANCED_BIN_ABST_V2_URLS,
            file_map=BALANCED_BIN_ABST_V2_FILE_MAP,
            citation=_VQA_BALANCED_BIN_ABST_CITATION,
            features=DEFAULT_FEATURES,
            name="balanced.binary.abstract",
            version=_VERSION,
        ),        
        VQAConfig(
            data_urls=BALANCED_REAL_COMP_PAIRS_URLS,
            file_map=BALANCED_REAL_COMP_PAIRS_FILE_MAP,
            citation=_VQA_V2_CITATION,
            features=COMP_PAIRS_FEATURES,
            name="complementary.pairs",
            version=_VERSION,
        ),
        VQAConfig(
            data_urls=V1_URLS,
            file_map=V1_MC_FILE_MAP,
            citation=_VQA_V1_CITATION,
            features=MC_FEATURES,
            name="v1.mc",
            version=_VERSION,
        ),
    ]

    BUILDER_CONFIG_CLASS = VQAConfig
    DEFAULT_CONFIG_NAME = "v2"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dl_dir = dl_manager.download_and_extract(self.config.data_urls)
        
        if "test" in dl_dir:
            split_kwargs = {
                datasets.Split.TRAIN: [dl_dir["train"], self.config.file_map["train"]],
                datasets.Split.VALIDATION: [dl_dir["validation"], self.config.file_map["validation"]],
                datasets.Split.TEST: [dl_dir["test"], self.config.file_map["test"]],
            }
        else:
            split_kwargs = {
                datasets.Split.TRAIN: [dl_dir["train"], self.config.file_map["train"]],
                datasets.Split.VALIDATION: [dl_dir["validation"], self.config.file_map["validation"]],
            }
        return [
            datasets.SplitGenerator(
                name=k, 
                gen_kwargs={
                    'extracted_files': v,
                }) for k, v in split_kwargs.items()
        ]

    def _generate_examples(self, extracted_files):
        """Yields examples."""
        extracted_path, file_names = extracted_files
        joined_extracted_path = {k:[os.path.join(p, f) for p, f in zip(extracted_path[k], file_names[k])] for k in extracted_path.keys()}

        if ".mc" in self.config.name:
            gen = functools.partial(generator, is_mc=True)
        elif "balanced.binary.abstract" in self.config.name:
            # split_str = "val"
            # answer_list, answer_confidence_list, multiple_choice_list = get_dicts_for_label(joined_extracted_path)
            # with open(f"./keti_air_vqa/balanced/{split_str}/answer_list.json", "w") as f:
            #     f.write(json.dumps(answer_list, indent=4))
            # with open(f"./keti_air_vqa/balanced/{split_str}/answer_confidence_list.json", "w") as f:
            #     f.write(json.dumps(answer_confidence_list, indent=4))
            # with open(f"./keti_air_vqa/balanced/{split_str}/multiple_choice_list.json", "w") as f:
            #     f.write(json.dumps(multiple_choice_list, indent=4))
            gen = functools.partial(generator, is_mc=False, convert2rgb=True)
        elif "complementary.pairs" in self.config.name:
            gen = generator_for_comp_pairs
        else:
            gen = functools.partial(generator, is_mc=False)

        for idx, item in enumerate(gen(joined_extracted_path)):
            yield idx, item



if __name__ == "__main__":
    from datasets import load_dataset

    raw_datasets = load_dataset(
        "keti_air_vqa.py",
        "balanced.binary.abstract.dd",
        cache_dir="huggingface_datasets",
        ignore_verifications=True,
        )
    dataset_train = raw_datasets["train"]
    for idx, item in enumerate(dataset_train):
        print(item)
        exit()