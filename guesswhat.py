from __future__ import annotations
import enum
import os
import glob
import json
import copy
from pyexpat import features
from stat import filemode
import textwrap
import logging
import zipfile
import functools
import re
from PIL import Image
import datasets

logger = logging.getLogger(__name__)

_VERSION = datasets.Version("0.1.0", "")

_URL = "https://github.com/GuessWhatGame/guesswhat"

_CITATION = """\
@inproceedings{guesswhat_game,
author = {Harm de Vries and Florian Strub and Sarath Chandar and Olivier Pietquin and Hugo Larochelle and Aaron C. Courville},
title = {GuessWhat?! Visual object discovery through multi-modal dialogue},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017}
}
@inproceedings{strub2017end,
  title={End-to-end optimization of goal-driven and visually grounded dialogue systems},
  author={Strub, Florian and De Vries, Harm and Mary, Jeremie and Piot, Bilal and Courville, Aaron and Pietquin, Olivier},
  booktitle={Proceedings of international joint conference on artificial intelligenc (IJCAI)},
  year={2017}
}
"""
_DESCRIPTION="""\
# GuessWhat?!

## Usage
```python
from datasets import load_dataset

raw_datasets = load_dataset(
    "guesswhat.py",
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
"""

# data path
GW_TRAIN = "https://florian-strub.com/guesswhat.train.jsonl.gz"
GW_VAL = "https://florian-strub.com//guesswhat.valid.jsonl.gz"
GW_TEST = "https://florian-strub.com//guesswhat.test.jsonl.gz"
COCO_TRAIN = "http://images.cocodataset.org/zips/train2014.zip"
COCO_VAL = "http://images.cocodataset.org/zips/val2014.zip"

BASE_URLS = {
    'GW_train_url': GW_TRAIN,
    'GW_validation_url': GW_VAL,
    'GW_test_url': GW_TEST,
    'COCO_train_url': COCO_TRAIN,
    'COCO_validation_url': COCO_VAL,
}
_BASE_IMAGE_URLS = {
    COCO_TRAIN: "train2014",
    COCO_VAL: "val2014",
}

BASE_FEATURES = datasets.Features(
    {
        "status": datasets.Value("string"), # "success"
        "picture": {
            "file_name": datasets.Value("string"),  # "COCO_val2014_000000534127.jpg",
            "flickr_url": datasets.Value("string"),  # "http://farm4.staticflickr.com/3187/2716894784_982b56503f_z.jpg",
            "width": datasets.Value("int32"),  # 640,
            "coco_url": datasets.Value("string"),  # "http://mscoco.org/images/534127",
            "height": datasets.Value("int32"),  # 360,
        },
        "picture_id": datasets.Value("int32"),  # 534127,
        "qas": datasets.Sequence({
            "q": datasets.Value("string"),  # "Is it a person?"
            "a": datasets.Value("string"),  # "Yes"
            "id": datasets.Value("int32"),  # 4967
        }),
        "questioner_id": datasets.Value("int32"),  # 0,
        "timestamp": datasets.Value("string"),  # "2016-07-08 15:06:34",
        "object_id": datasets.Value("int32"),  # 2159052,
        "dialogue_id": datasets.Value("int32"),  # 2416,
        "objects": datasets.Sequence({
                "category": datasets.Value("string"), # "person",
                "area": datasets.Value("float"),  # 2487.0305500000004, 
                "iscrowd": datasets.Value("bool"),  # false, 
                "object_id": datasets.Value("int32"),  # 480437, 
                "bbox": datasets.Sequence(datasets.Value("float32"), length=4),  # [21.08, 145.54, 47.84, 94.05]
                "category_id": datasets.Value("int32"),  # 1, 
                "segment": datasets.Sequence(datasets.Sequence(datasets.Value("float32"),)),  # [[39.73, 229.86, 43.78, 222.57, 38.11, 211.22, 34.86, 213.65, 27.57, 209.59, 25.14, 196.62, 21.08, 177.97, 33.24, 161.76, 29.19, 153.65, 33.24, 145.54, 48.65, 147.97, 48.65, 159.32, 48.65, 163.38, 56.76, 176.35, 64.86, 189.32, 68.92, 190.14, 64.05, 202.3, 62.43, 210.41, 57.57, 217.7, 66.49, 234.73, 60.0, 239.59]]
        }),
        "image": datasets.Image(),  # PIL.open(),
    }
)
# License: Creative Commons Attribution 4.0 International License
'''
{
    'status': 'success',
    'picture': {
        'file_name': 'COCO_val2014_000000534127.jpg',
        'flickr_url': 'http://farm4.staticflickr.com/3187/2716894784_982b56503f_z.jpg',
        'width': 640,
        'coco_url': 'http://mscoco.org/images/534127',
        'height': 360
    },
    'picture_id': 534127,
    'qas': [{
            'q': 'Is it a person?',
            'a': 'Yes',
            'id': 4967
        },{
            'q': 'Is the person a boy?',
            'a': 'Yes',
            'id': 4969
    }],
    'questioner_id': 0,
    'timestamp': '2016-07-08 15:06:34',
    'object_id': 2159052,
    'dialogue_id': 2416,
    'objects': {
        '480437': {
            'category': 'person',
            'area': 2487.0305500000004,
            'iscrowd': False,
            'object_id': 480437,
            'bbox': [21.08, 145.54, 47.84, 94.05],
            'category_id': 1,
            'segment': [[39.73, 229.86, 43.78, 222.57, 38.11, 211.22, 34.86, 213.65, 27.57, 209.59, 25.14, 196.62, 21.08, 177.97, 33.24, 161.76, 29.19, 153.65, 33.24, 145.54, 48.65, 147.97, 48.65, 159.32, 48.65, 163.38, 56.76, 176.35, 64.86, 189.32, 68.92, 190.14, 64.05, 202.3, 62.43, 210.41, 57.57, 217.7, 66.49, 234.73, 60.0, 239.59]]
        }, '475656': {
            'category': 'person', 'area': 1153.1943000000003, 'iscrowd': False, 'object_id': 475656, 'bbox': [214.2, 131.16, 39.21, 74.96], 'category_id': 1, 'segment': [[233.04, 133.47, 230.34, 134.62, 228.81, 136.55, 228.42, 141.16, 225.35, 143.85, 222.66, 146.54, 215.74, 154.61, 214.2, 159.61, 217.66, 168.84, 220.73, 168.84, 217.28, 161.92, 219.58, 158.07, 223.43, 154.61, 223.43, 172.68, 232.65, 189.98, 225.35, 200.36, 225.73, 203.43, 226.89, 204.58, 231.5, 198.43, 236.11, 192.67, 236.5, 190.75, 236.88, 185.36, 238.03, 187.29, 238.8, 196.9, 240.72, 206.12, 242.65, 204.58, 244.18, 196.9, 244.18, 188.44, 238.8, 169.6, 238.03, 161.53, 245.72, 168.84, 253.41, 169.22, 252.26, 165.38, 250.33, 163.84, 243.03, 157.3, 241.11, 147.69, 239.19, 142.7, 239.96, 135.01, 234.96, 131.16]]}, '1168669': {'category': 'backpack', 'area': 1438.2855, 'iscrowd': False, 'object_id': 1168669, 'bbox': [314.99, 173.04, 50.16, 65.6], 'category_id': 27, 'segment': [[363.67, 235.67, 349.42, 234.19, 342.89, 230.33, 339.03, 226.17, 346.16, 236.26, 342.59, 238.64, 336.06, 226.17, 327.46, 216.97, 323.6, 211.33, 318.85, 203.02, 314.99, 194.41, 315.29, 183.13, 320.33, 178.97, 323.6, 173.33, 327.46, 173.04, 330.42, 181.65, 336.95, 183.13, 341.7, 183.72, 344.67, 184.32, 350.61, 186.99, 357.44, 191.44, 360.11, 193.22, 362.48, 197.08, 363.37, 200.64, 365.15, 207.17, 365.15, 208.36, 364.86, 208.95, 359.51, 206.58, 358.03, 204.8, 355.95, 200.64, 353.28, 196.78, 346.45, 192.63, 337.84, 192.63, 336.66, 195.3, 339.92, 205.99, 343.78, 211.92, 347.64, 215.78, 350.31, 217.86]]}, '603633': {'category': 'frisbee', 'area': 853.7405999999996, 'iscrowd': False, 'object_id': 603633, 'bbox': [415.86, 146.27, 37.02, 32.13], 'category_id': 34, 'segment': [[437.98, 175.14, 446.82, 170.02, 450.32, 166.29, 452.41, 160.47, 452.88, 156.28, 451.71, 151.86, 450.32, 148.83, 446.36, 146.5, 442.4, 146.27, 436.58, 146.27, 428.43, 149.53, 421.44, 156.05, 417.25, 162.8, 415.86, 167.69, 417.72, 173.28, 423.07, 177.7, 430.53, 178.4, 440.3, 173.74]]}, '2159052': {'category': 'person', 'area': 7317.549849999998, 'iscrowd': False, 'object_id': 2159052, 'bbox': [322.08, 149.56, 99.72, 173.88], 'category_id': 1, 'segment': [[322.08, 165.6, 334.61, 151.57, 342.63, 149.56, 349.64, 153.57, 355.15, 161.09, 358.66, 162.59, 360.67, 162.59, 365.18, 162.59, 372.19, 163.09, 390.73, 184.64, 406.27, 202.68, 419.8, 209.19, 420.8, 217.71, 407.77, 216.21, 395.24, 199.17, 388.23, 199.17, 398.75, 213.7, 403.76, 221.22, 410.27, 221.72, 420.8, 223.22, 421.8, 224.72, 421.8, 232.24, 414.78, 236.75, 415.79, 239.26, 421.3, 267.82, 408.27, 283.35, 402.76, 289.37, 398.25, 301.39, 396.24, 304.4, 402.26, 311.91, 392.74, 319.43, 382.71, 323.44, 373.19, 317.93, 378.2, 282.85, 386.72, 279.34, 383.72, 266.32, 379.21, 253.29, 376.2, 246.27, 369.19, 237.75, 362.17, 233.24, 351.15, 220.22, 342.63, 212.7, 333.61, 199.17, 337.62, 191.65, 331.6, 186.14, 335.11, 181.63, 331.1, 173.61]]}}}

'''

class GuessWhatConfig(datasets.BuilderConfig):
    """BuilderConfig for VGDetection"""

    def __init__(
        self,
        data_urls=BASE_URLS,
        citation=_CITATION,
    **kwargs):
        super(GuessWhatConfig, self).__init__(**kwargs)
        self.data_urls = data_urls
        self.citation = citation
    
    @property
    def features(self):
        return BASE_FEATURES

class GuessWhat(datasets.GeneratorBasedBuilder):
    """VGDetection"""
    BUILDER_CONFIGS = [
        GuessWhatConfig(
            data_urls=BASE_URLS,
            citation=_CITATION,
            name="base",
            version=_VERSION,
        ),
    ]

    BUILDER_CONFIG_CLASS = GuessWhatConfig
    DEFAULT_CONFIG_NAME = "base"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features = self.config.features,
            supervised_keys=None,
            homepage=_URL,
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dl_dir = dl_manager.download_and_extract(self.config.data_urls)
        # print(dl_dir)
        """
        {'GW_train_url': 'huggingface_datasets/downloads/extracted/718d812f30c3d20077b27710c12e9f0d454ca604160df4bec861f77e51b8de1f',
        'GW_validation_url': 'huggingface_datasets/downloads/extracted/ae8827e2ff874de9abe6c9b0891be72051fe2ee1be55075b87e4bc994f52d44b',
        'GW_test_url': 'huggingface_datasets/downloads/extracted/c47fe2405f0717795b76fe9b35e1f3bc12b8f891b700755c2e5bbf257384707c',
        'COCO_train_url': 'huggingface_datasets/downloads/extracted/f43997a6b702c171736e1483130b42a1020f67a32525a867b9b3129642d1daa3',
        'COCO_validation_url': 'huggingface_datasets/downloads/extracted/5ad45e7951d03e24a8252b2cbac7382000a0b4a80f921ae2a7c520adde784938'
        }
        """
        # rewire -> extract only number in files
        # md5sum
        # get another file (feature) or preprocess
        # get dict.  
        split_kwargs = {
                datasets.Split.TRAIN: [dl_dir["GW_train_url"], dl_dir["COCO_train_url"], dl_dir["COCO_validation_url"]],
                datasets.Split.VALIDATION: [dl_dir["GW_validation_url"], dl_dir["COCO_train_url"], dl_dir["COCO_validation_url"]],
                datasets.Split.TEST: [dl_dir["GW_test_url"], dl_dir["COCO_train_url"], dl_dir["COCO_validation_url"]],
            }
        return [
            datasets.SplitGenerator(
                name=k,
                gen_kwargs={
                    'extracted_files': v,
                }) for k, v in split_kwargs.items()
        ]

    def _generate_examples(self, **kwargs):
        # print(kwargs)  # {'extracted_files': ['huggingface_datasets/downloads/extracted/718d812f30c3d20077b27710c12e9f0d454ca604160df4bec861f77e51b8de1f', 'huggingface_datasets/downloads/extracted/f43997a6b702c171736e1483130b42a1020f67a32525a867b9b3129642d1daa3', 'huggingface_datasets/downloads/extracted/5ad45e7951d03e24a8252b2cbac7382000a0b4a80f921ae2a7c520adde784938']}
        """Yields examples."""
        with open(kwargs['extracted_files'][0], "r", encoding="utf-8") as f:
            # print(f)
            for idx, line in enumerate(f):
                item = json.loads(line)
                # print(loaded_line)
                dict_to_list = []
                for k, v in item["objects"].items():
                    dict_to_list.append(v)
                item["objects"] = dict_to_list
                file_name = item["picture"]["file_name"]  # COCO_val2014_*.jpg or COCO_train2014_*.jpg
                folder_name = glob.glob(kwargs['extracted_files'][1]+"/*")[0] if 'train' in file_name else glob.glob(kwargs['extracted_files'][2]+"/*")[0]
                # print(folder_name, file_name)
                # item["image"] = Image.open(os.path.join(folder_name, file_name)).convert("RGB")
                item["image"] = os.path.join(folder_name, file_name)
                yield idx, item



if __name__ == "__main__":
    from datasets import load_dataset

    raw_datasets = load_dataset(
        "guesswhat.py",
        "base",
        cache_dir="huggingface_datasets",
        data_dir="data",
        ignore_verifications=True,
        )
    dataset_train = raw_datasets["train"]

    for item in dataset_train:
        print(item)
        exit()