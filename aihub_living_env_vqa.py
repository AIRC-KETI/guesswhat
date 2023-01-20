import os
import glob
import json
import textwrap
import logging
import zipfile
import functools

from tqdm import tqdm
import datasets

logger = logging.getLogger(__name__)

_VERSION = datasets.Version("1.0.0", "")

_URL = "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=101"

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
# 생활 및 거주환경 기반 VQA

## 소개
(대전시 유성구)국내 환경에 맞는 다양한 VQA 기반 AI서비스 개발을 위한 생활 및 거주환경 VQA AI데이터

## 구축목적
- 어린이, 노인, 개인의 일상생활을 촬영한 이미지에 대하여 시각정보에 대한 객관적인 상황이나 추론 가능한 질문에 대해 스스로 답변이 가능한 인공지능을 훈련하기 위한 데이터 셋

## 활용분야
- 시각 정보에 대한 인공지능 자유 묘사, 이미지를 통한 상황 유추 등이 가능한 한국형 AI 시각지능 모델 개발

## 소개
- 한국인의 실생활 속에서 다양한 이미지를 촬영하고, 연관된 질의응답 데이터를 생성하여 인공지능이 생활환경 속 물체나 위험요소 등에 대하여 답변할 수 있도록 훈련할 수 있는 데이터셋. 이미지에 대한 비식별화 및 정제 처리 후 가공, 검증을 진행하여 촬영된 사진에서 개인정보 침해 문제를 해결하고 가공을 수행하였음

## 구축 내용 및 제공 데이터량
- 일상생활 속 이미지 1,063,340장(일반 촬영 961,068장 / 3D 공간 스캔 기반 추출 이미지 102,272장)
- 이미지별 질의응답 텍스트 총 7,119,756건(이미지당 평균 7건)

## Usage
```python
from datasets import load_dataset

raw_datasets = load_dataset(
                "aihub_living_env_vqa.py", 
                "default",
                cache_dir="huggingface_datasets", 
                data_dir="data",
                ignore_verifications=True,
            )

dataset_train = raw_datasets["train"]

for item in dataset_train:
    print(item)
    exit()
```

## 데이터 관련 문의처
| 담당자명 | 전화번호 | 이메일 |
| ------------- | ------------- | ------------- |
| 나현우(유클리드소프트) | 042-488-6589 | hwna@euclidsoft.co.kr |

## Copyright

### 데이터 소개
AI 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI 응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.
본 AI데이터 등은 인공지능 기술 및 제품·서비스 발전을 위하여 구축하였으며, 지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다.

### 데이터 이용정책
- 본 AI데이터 등을 이용하기 위해서 다음 사항에 동의하며 준수해야 함을 고지합니다.

1. 본 AI데이터 등을 이용할 때에는 반드시 한국지능정보사회진흥원의 사업결과임을 밝혀야 하며, 본 AI데이터 등을 이용한 2차적 저작물에도 동일하게 밝혀야 합니다.
2. 국외에 소재하는 법인, 단체 또는 개인이 AI데이터 등을 이용하기 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가 필요합니다.
3. 본 AI데이터 등의 국외 반출을 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가 필요합니다.
4. 본 AI데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 한국지능정보사회진흥원은 AI데이터 등의 이용의 목적이나 방법, 내용 등이 위법하거나 부적합하다고 판단될 경우 제공을 거부할 수 있으며, 이미 제공한 경우 이용의 중지와 AI 데이터 등의 환수, 폐기 등을 요구할 수 있습니다.
5. 제공 받은 AI데이터 등을 수행기관 등과 한국지능정보사회진흥원의 승인을 받지 않은 다른 법인, 단체 또는 개인에게 열람하게 하거나 제공, 양도, 대여, 판매하여서는 안됩니다.
6. AI데이터 등에 대해서 제 4항에 따른 목적 외 이용, 제5항에 따른 무단 열람, 제공, 양도, 대여, 판매 등의 결과로 인하여 발생하는 모든 민・형사 상의 책임은 AI데이터 등을 이용한 법인, 단체 또는 개인에게 있습니다.
7. 이용자는 AI 허브 제공 데이터셋 내에 개인정보 등이 포함된 것이 발견된 경우, 즉시 AI 허브에 해당 사실을 신고하고 다운로드 받은 데이터셋을 삭제하여야 합니다.
8. AI 허브로부터 제공받은 비식별 정보(재현정보 포함)를 인공지능 서비스 개발 등의 목적으로 안전하게 이용하여야 하며, 이를 이용해서 개인을 재식별하기 위한 어떠한 행위도 하여서는 안됩니다.
9. 향후 한국지능정보사회진흥원에서 활용사례・성과 등에 관한 실태조사를 수행 할 경우 이에 성실하게 임하여야 합니다.

### 데이터 다운로드 신청방법
1. AI 허브를 통해 제공 중인 AI데이터 등을 다운로드 받기 위해서는 별도의 신청자 본인 확인과 정보 제공, 목적을 밝히는 절차가 필요합니다.
2. AI데이터를 제외한 데이터 설명, 저작 도구 등은 별도의 신청 절차나 로그인 없이 이용이 가능합니다.
3. 한국지능정보사회진흥원이 권리자가 아닌 AI데이터 등은 해당 기관의 이용정책과 다운로드 절차를 따라야 하며 이는 AI 허브와 관련이 없음을 알려 드립니다.

"""

# training data path
TRAINING_ROOT_PATH_REL = "생활 및 거주환경 기반 VQA/Training"

TRAINING_LBLS_PERSONAL_FPATH_REL = [("[라벨]개인생활환경 이미지 수집.zip", "개인생활환경 이미지 수집")]
TRAINING_LBLS_OLDMAN_FPATH_REL = [("[라벨]노인 생활거주환경 이미지 수집.zip", "노인 생활거주환경 이미지 수집")]
TRAINING_LBLS_INDOOR_FPATH_REL = [
    ("[라벨]실내 가전 및 가구배치 이미지 수집_1.zip", "실내 가전 및 가구배치 이미지 수집"), 
    ("[라벨]실내 가전 및 가구배치 이미지 수집_2.zip", "실내 가전 및 가구배치 이미지 수집")
]
TRAINING_LBLS_CHILDREN_FPATH_REL = [("[라벨]어린이 생활거주환경 이미지 수집.zip", "어린이 생활거주환경 이미지 수집")]
TRAINING_LBLS_ALL_REL = TRAINING_LBLS_PERSONAL_FPATH_REL + TRAINING_LBLS_OLDMAN_FPATH_REL + TRAINING_LBLS_INDOOR_FPATH_REL + TRAINING_LBLS_CHILDREN_FPATH_REL

TRAINING_IMGS_PERSONAL_FPATH_REL = ["개인생활환경 이미지 수집.zip"]
TRAINING_IMGS_OLDMAN_FPATH_REL = ["노인 생활거주환경 이미지 수집.zip"]
TRAINING_IMGS_INDOOR_FPATH_REL = ["실내 가전 및 가구배치 이미지 수집_1.zip", "실내 가전 및 가구배치 이미지 수집_2.zip"]
TRAINING_IMGS_CHILDREN_FPATH_REL = ["어린이 생활거주환경 이미지 수집.zip"]
TRAINING_IMGS_ALL_REL = TRAINING_IMGS_PERSONAL_FPATH_REL + TRAINING_IMGS_OLDMAN_FPATH_REL + TRAINING_IMGS_INDOOR_FPATH_REL + TRAINING_IMGS_CHILDREN_FPATH_REL

# validation data path
VALIDATION_ROOT_PATH_REL = "생활 및 거주환경 기반 VQA/Validation"

VALIDATION_LBLS_PERSONAL_FPATH_REL = [("[라벨]개인생활환경 이미지 수집.zip", "개인생활환경 이미지 수집")]
VALIDATION_LBLS_OLDMAN_FPATH_REL = [("[라벨]노인 생활거주환경 이미지 수집.zip", "노인 생활거주환경 이미지 수집")]
VALIDATION_LBLS_INDOOR_FPATH_REL = [("[라벨]실내 가전 및 가구배치 이미지 수집.zip", "실내 가전 및 가구배치 이미지 수집")]
VALIDATION_LBLS_CHILDREN_FPATH_REL = [("[라벨]어린이 생활거주환경 이미지 수집.zip", "어린이 생활거주환경 이미지 수집")]
VALIDATION_LBLS_ALL_REL = VALIDATION_LBLS_PERSONAL_FPATH_REL + VALIDATION_LBLS_OLDMAN_FPATH_REL + VALIDATION_LBLS_INDOOR_FPATH_REL + VALIDATION_LBLS_CHILDREN_FPATH_REL

VALIDATION_IMGS_PERSONAL_FPATH_REL = ["개인생활환경 이미지 수집.zip"]
VALIDATION_IMGS_OLDMAN_FPATH_REL = ["노인 생활거주환경 이미지 수집.zip"]
VALIDATION_IMGS_INDOOR_FPATH_REL = ["실내 가전 및 가구배치 이미지 수집.zip"]
VALIDATION_IMGS_CHILDREN_FPATH_REL = ["어린이 생활거주환경 이미지 수집.zip"]
VALIDATION_IMGS_ALL_REL = VALIDATION_IMGS_PERSONAL_FPATH_REL + VALIDATION_IMGS_OLDMAN_FPATH_REL + VALIDATION_IMGS_INDOOR_FPATH_REL + VALIDATION_IMGS_CHILDREN_FPATH_REL



def check_extraction(root_dir, out_dir, file_list):
    output_dir = os.path.join(root_dir, out_dir)
    try:
        for fpath_rel in file_list:
            base_name, _ = os.path.splitext(os.path.basename(fpath_rel))
            os.makedirs(output_dir, exist_ok=True)

            if os.path.isdir(os.path.join(output_dir, base_name)):
                logger.info("The files are alread extracted: {}".format(base_name))
                continue
            
            fpath = os.path.join(root_dir, fpath_rel)
            logger.info("Extracting {} files...".format(fpath))
            with zipfile.ZipFile(fpath, "r") as fp:
                fp.extractall(output_dir)
    except Exception as e:
        logger.info("The output directory({}) already exists.".format(output_dir))
        logger.info(e)
        pass

# equivalent with check_extraction
def check_extraction2(root_dir, out_dir, file_list):
    output_dir = os.path.join(root_dir, out_dir)
    try:
        os.makedirs(output_dir)
        for fpath_rel in file_list:
            fpath = os.path.join(root_dir, fpath_rel)
            with zipfile.ZipFile(fpath, "r") as fp:
                flist = fp.namelist()
                flist = list(filter(lambda x: x.endswith(".jpg") or x.endswith(".jpeg"), flist))
                logger.info("Extracting {} files for {}...".format(len(flist), fpath_rel))

                for jpg_file in tqdm(flist, desc="{}".format(fpath_rel)):
                    fp.extract(jpg_file, output_dir)
    except Exception as e:
        logger.info("The output directory({}) already exists.".format(output_dir))
        logger.info(e)
        pass

def parsing_item(img_root, fpath):
    
    with zipfile.ZipFile(fpath, "r") as fp:
        flist = fp.namelist()
        flist = filter(lambda x: x.endswith(".json"), flist)
        for fname in flist:
            item_list = json.load(fp.open(fname, "r"))
            images = item_list["images"]
            qas = item_list["question"]
            images_dict = {k["image_id"]: k for k in images}

            for qa in qas:
                image_id = qa["image_id"]
                qa_image = images_dict.get(image_id, None)
                if qa_image is None:
                    continue
                image_fname = qa_image["image"]

                yield os.path.join(img_root, qa_image["category"], image_fname), {
                    "image_info":{
                        "image_id": image_id,
                        "image_fname": image_fname,
                        "category": qa_image["category"],
                        "weather": qa_image["weather"],
                    },
                    "question_id": qa["question_id"],
                    "question": qa["question"],
                    "answers": [
                        {
                            "answer": qa["answer"],
                        }
                    ],
                    "answer_type": qa["answer_type"],
                    "multiple_choice_answer": qa["answer"],
                }

def generator(data_root, fpath_list):
    idx = 0
    for fpath, sub_dir in fpath_list:
        for image_path, item in parsing_item(
            os.path.join(data_root, "images", sub_dir), 
            os.path.join(data_root, fpath)):
            item["image"] = {
                    "path": image_path,
                    "bytes": open(image_path, "rb").read(),
                }
            yield idx, item
            idx += 1


DEFAULT_FEATURES=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_info": datasets.Features({
                        "image_id": datasets.Value("string"),
                        "image_fname": datasets.Value("string"),
                        "category": datasets.Value("string"),
                        "weather": datasets.Value("string"),
                    }),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence({
                        "answer": datasets.Value("string"),
                    }),
                    "answer_type": datasets.Value("string"),
                    "multiple_choice_answer": datasets.Value("string"),
                }
            )

class LivingEnvVQAConfig(datasets.BuilderConfig):
    """BuilderConfig for LivingEnvVQA."""

    def __init__(
            self, 
            training_lbl_path=TRAINING_LBLS_ALL_REL, 
            validation_lbl_path=VALIDATION_LBLS_ALL_REL, 
            **kwargs):
        """BuilderConfig for LivingEnvVQA.

        Args:
        features: datasets.Feature for the dataset.
        **kwargs: keyword arguments forwarded to super.
        """
        super(LivingEnvVQAConfig, self).__init__(**kwargs)
        self.training_lbl_path = training_lbl_path
        self.validation_lbl_path = validation_lbl_path


class LivingEnvVQA(datasets.GeneratorBasedBuilder):
    """LivingEnvVQA Dataset"""

    BUILDER_CONFIGS = [
        LivingEnvVQAConfig(
            training_lbl_path=TRAINING_LBLS_ALL_REL,
            validation_lbl_path=VALIDATION_LBLS_ALL_REL,
            name="all",
            version=_VERSION,
        ),
        LivingEnvVQAConfig(
            training_lbl_path=TRAINING_LBLS_PERSONAL_FPATH_REL,
            validation_lbl_path=VALIDATION_LBLS_PERSONAL_FPATH_REL,
            name="personal",
            version=_VERSION,
        ),
        LivingEnvVQAConfig(
            training_lbl_path=TRAINING_LBLS_OLDMAN_FPATH_REL,
            validation_lbl_path=VALIDATION_LBLS_OLDMAN_FPATH_REL,
            name="oldman",
            version=_VERSION,
        ),
        LivingEnvVQAConfig(
            training_lbl_path=TRAINING_IMGS_INDOOR_FPATH_REL,
            validation_lbl_path=VALIDATION_LBLS_INDOOR_FPATH_REL,
            name="indoor",
            version=_VERSION,
        ),
        LivingEnvVQAConfig(
            training_lbl_path=TRAINING_LBLS_CHILDREN_FPATH_REL,
            validation_lbl_path=VALIDATION_LBLS_CHILDREN_FPATH_REL,
            name="children",
            version=_VERSION,
        ),
    ]

    BUILDER_CONFIG_CLASS = LivingEnvVQAConfig
    DEFAULT_CONFIG_NAME = "all"

    manual_download_instructions = textwrap.dedent(f"""
        You need to manually download the data file on AIHub (${_URL}). 
        The folder containing the saved file can be used to load the dataset 
        via 'datasets.load_dataset("aihub_living_env_vqa.py", data_dir="<path/to/folder>")'
    """)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=DEFAULT_FEATURES,
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        EXTRACTION_CHECK_LIST = [
            (os.path.join(dl_manager.manual_dir, TRAINING_ROOT_PATH_REL), "images", TRAINING_IMGS_ALL_REL),
            (os.path.join(dl_manager.manual_dir, VALIDATION_ROOT_PATH_REL), "images", VALIDATION_IMGS_ALL_REL),
        ]
        for root_path, out_dir, pattern in EXTRACTION_CHECK_LIST:
            check_extraction2(root_path, out_dir, pattern)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    'manual_dir_path': dl_manager.manual_dir,
                    'root_path_rel': TRAINING_ROOT_PATH_REL,
                    'fpath_list': self.config.training_lbl_path
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={
                    'manual_dir_path': dl_manager.manual_dir,
                    'root_path_rel': VALIDATION_ROOT_PATH_REL,
                    'fpath_list': self.config.validation_lbl_path
                })
        ]

    def _generate_examples(self, manual_dir_path, root_path_rel, fpath_list):
        """Yields examples."""

        data_root = os.path.join(manual_dir_path, root_path_rel)

        for idx, item in generator(data_root, fpath_list):
            yield idx, item


if __name__ == "__main__":
    from datasets import load_dataset

    raw_datasets = load_dataset(
        "aihub_living_env_vqa.py",
        "all.dd",
        cache_dir="huggingface_datasets",
        ignore_verifications=True,
        )
    dataset_train = raw_datasets["train"]
    for idx, item in enumerate(dataset_train):
        print(item)
        exit()