import os
import functools

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import (
    AutoConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoProcessor,
    AutoFeatureExtractor,
    DataCollatorWithPadding,
)

from models.fusion.configuration_fusion import FusionConfig
from models.oracle.configuration_oracle import OracleConfig
from models.oracle.modeling_oracle import OracleModel


fusion_model_config_path = "models/fusion/config.json"
vision_model_path = "openai/clip-vit-base-patch32"
language_model_path = "roberta-base"
output_dir = "test_model"

per_device_train_batch_size = 2

fusion_model_config = FusionConfig.from_pretrained(fusion_model_config_path)
vision_config = AutoConfig.from_pretrained(vision_model_path)
text_config = AutoConfig.from_pretrained(language_model_path)
vision_text_model_config = VisionTextDualEncoderConfig.from_vision_text_configs(
    vision_config,
    text_config
)
oracle_config = OracleConfig.from_vision_text_fusion_configs(
    vision_text_model_config=vision_text_model_config, 
    fusion_model_config=fusion_model_config
)



model = OracleModel(oracle_config, vision_model_path=vision_model_path, language_model_path=language_model_path)
# model = OracleModel.from_pretrained(output_dir)
# model = OracleModel(oracle_config)

raw_datasets = load_dataset(
        "guesswhat.py",
        "oracle",
        cache_dir="huggingface_datasets"
    )
raw_datasets.set_format(type='torch')
raw_datasets = raw_datasets.remove_columns("question_id")
features = raw_datasets["train"].features
num_labels = features["answer"].num_classes

id2label={ i:features["answer"].int2str(i) for i in range(num_labels)}
label2id={ c:i for i, c in id2label.items()}

num_labels_cat = features["category"].num_classes
id2label_cat={ i:features["category"].int2str(i) for i in range(num_labels_cat)}
label2id_cat={ c:i for i, c in id2label_cat.items()}

tokenizer = AutoTokenizer.from_pretrained(language_model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(vision_model_path)
processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)


image_size = vision_text_model_config.vision_config.image_size

def calc_bbox(wh, bbox, size=224):
    
    width, height = wh
    x, y, w, h = bbox
    # print("w: {}, h: {}".format(width, height))
    # print("x, y, w, h: {}, {}, {}, {}".format(x, y, w, h))
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size if isinstance(size, int) else size[0]
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    if width <= height:
        nx, nw = x, w
        top = (new_long - size)//2
        bottom = top + size
        y = y * new_long
        h = h * new_long
        ny = min(max(y - top, 0), size)/size
        nh = min(max(h + y - top, 0)/size - ny, size)
    else:
        ny, nh = y, h
        left = (new_long - size)//2
        right = left + size
        x = x * new_long
        w = w * new_long

        nx = min(max(x - left, 0), size)/size
        nw = min(max(w + x - left, 0)/size - nx, size)
    # x, y, w, h = resized absolute position
    
    return (nx, ny, nw, nh)

calc_bbox_fn = functools.partial(calc_bbox, size=image_size)

def encode(features):
    features_tmp = processor(text=features["question"], images=[image.convert("RGB") for image in features["image"]], return_tensors="pt")
    wh_l = [image.size for image in features["image"]]
    bbox_l = [bbox for bbox in features["bbox"]]
    # print(bbox_l)
    
    features_tmp["bbox"] = [calc_bbox_fn(wh, bbox) for wh, bbox in zip(wh_l, bbox_l)]
    # print(features_tmp["bbox"])
    
    for k in ["answer", "category"]:
        features_tmp[k] = features[k]
    return features_tmp

raw_datasets.set_transform(encode)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]
data_collator = DataCollatorWithPadding(processor.tokenizer, pad_to_multiple_of=None)
train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=per_device_train_batch_size)


# feature_extractor_p = processor.current_processor
image_mean = torch.FloatTensor(feature_extractor.image_mean)
image_std = torch.FloatTensor(feature_extractor.image_std)
image_size = vision_text_model_config.vision_config.image_size

for idx, batch in zip(range(1), train_dataloader):
    outputs = model(**batch)



import numpy as np
import PIL.Image
import PIL.ImageDraw


c = (255, 0, 0)

for idx, batch in zip(range(3), train_dataloader):

    for sid in range(per_device_train_batch_size):
        image = batch["pixel_values"][sid]*image_std[:,None,None] + image_mean[:,None,None]
        image = image.numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        image*=255.
        image = image.astype(np.uint8)
        image = PIL.Image.fromarray(image)


        bb = batch["bbox"][sid].numpy()*image_size
        print(bb)
        img = image.convert("RGBA")
        overlay = PIL.Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(overlay)
        draw.rectangle((bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]), outline=c, width = 3)
        img = PIL.Image.alpha_composite(img, overlay)
        image = img.convert("RGB")

        answer = batch["answer"][sid].numpy()
        answer = id2label[int(answer)]

        cat = batch["category"][sid].numpy()
        cat = id2label_cat[int(cat)]


        image.save(os.path.join(output_dir, f"{idx}_{sid}.jpg"))
        print(tokenizer.decode(batch["input_ids"][sid]), answer, cat)


# for idx, item in zip(range(6), train_dataset):
#     image = item["image"]
#     print(image)


#     bb = item["bbox"]
#     w, h = image.size
#     img = image.convert("RGBA")
#     overlay = PIL.Image.new('RGBA', img.size, (0, 0, 0, 0))
#     draw = PIL.ImageDraw.Draw(overlay)
#     draw.rectangle((bb[0]*w, bb[1]*h, (bb[0]+bb[2])*w, (bb[1]+bb[3])*h), outline=c, width = 3)
#     img = PIL.Image.alpha_composite(img, overlay)
#     image = img.convert("RGB")

#     answer = item["answer"]
#     answer = id2label[int(answer)]

#     cat = item["category"]
#     cat = id2label_cat[int(cat)]


#     image.save(os.path.join(output_dir, f"{idx:03d}.jpg"))
#     print(item["question"], answer, cat)





# model.save_pretrained(output_dir)
# oracle_config.save_pretrained(output_dir)
