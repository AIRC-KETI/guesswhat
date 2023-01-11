import os
import argparse
import math
import random
import logging
import json
import functools
from typing import (
    Union, 
    Optional, 
    List, 
    Dict, 
    Any
)
from dataclasses import dataclass
from datetime import timedelta

import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from datasets.utils import logging as datasets_logging

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedDataParallelKwargs

from transformers.utils import logging as transformers_logging
from transformers.utils import PaddingStrategy
from transformers import (
    AutoConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderProcessor,
    AutoProcessor,
    AutoTokenizer,
    AutoFeatureExtractor,
    SchedulerType,
    get_scheduler,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    default_data_collator,
)
import evaluate

from models.fusion.configuration_fusion import FusionConfig
from models.oracle.configuration_oracle import OracleConfig
from models.oracle.oracle import OracleModelForSequenceClassification

KETIAIR_TOKEN = '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'
logger = get_logger(__name__)


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        batch = default_data_collator(batch)
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="")
    # data
    parser.add_argument(
        "--feature_extractor_name",
        type=str,
        default= None, 
        help="Pretrained feature extractor name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default= None, 
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default= None, 
        help="Pretrained processor name or path if not the same as model_name",
    )
    parser.add_argument(
        "--vision_language_model_path",
        type=str,
        default= None, 
        help="Pretrained vision language model name or path if not the same as model_name",
    )
    parser.add_argument(
        "--language_model_path",
        type=str,
        default= "bert-base-uncased", 
        help="Pretrained language model name or path if not the same as model_name",
    )
    parser.add_argument(
        "--vision_model_path",
        type=str,
        default= "google/vit-base-patch16-224", 
        help="Pretrained vision model name or path if not the same as model_name",
    )
    parser.add_argument(
        "--fusion_model_config_path",
        type=str,
        default= "models/fusion/config.json", 
        help="Pretrained fusion model name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default= None, 
        help="Pretrained model name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=32,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--with_category_info",
        action="store_true",
        help="Whether to use category information for classification.",
    )
    parser.add_argument(
        "--with_position_info",
        action="store_true",
        help="Whether to use the position information of target object for classification.",
    )
    parser.add_argument(
        "--norm_position",
        action="store_true",
        help="Whether to normalize the position information of target object",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    # scheduler
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1,
        help="Total number of training epochs to perform.")

    # checkpoint
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # training
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",type=int, default=64,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--max_train_steps_per_epoch", type=int, default=None,
        help="The number of training steps to perform on a epoch. (for debugging)",
    )
    parser.add_argument(
        "--warmup_portion", type=float, default=0, help="Portion of total training steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=8e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    # logging
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # initialize Accelerator
    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(days=10)),
        DistributedDataParallelKwargs(find_unused_parameters=True)
    ]
    
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        kwargs_handlers=kwargs_handlers , **accelerator_log_kwargs
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets_logging.set_verbosity_warning()
        transformers_logging.set_verbosity_info()
    else:
        datasets_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # dataset
    raw_datasets = load_dataset(
        "guesswhat.py",
        "oracle",
        cache_dir="huggingface_datasets"
    )
    # map cache, transform = on the fly, cache X
    # loader [sampler, batch_sampler], collate_fn <- (only batch_sampler's output)
    raw_datasets.set_format(type='torch')
    # remove question_id
    raw_datasets = raw_datasets.remove_columns("question_id")

    features = raw_datasets["train"].features
    num_labels = features["answer"].num_classes
    
    id2label={ i:features["answer"].int2str(i) for i in range(num_labels)}
    label2id={ c:i for i, c in id2label.items()}

    # Config Setting
    # model = OracleModel.from_pretrained("oracle", use_auth_token=KETIAIR_TOKEN)
    # vision_model_path=None, 
    # language_model_path=None,

    if args.model_name_or_path is not None:
        oracle_config = OracleConfig.from_pretrained(args.model_name_or_path)
    elif args.vision_language_model_path is not None:
        fusion_model_config = FusionConfig.from_pretrained(args.fusion_model_config_path)
        vision_text_model_config = VisionTextDualEncoderConfig.from_pretrained(args.vision_language_model_path)
        oracle_config = OracleConfig.from_vision_text_fusion_configs(
            vision_text_model_config=vision_text_model_config, 
            fusion_model_config=fusion_model_config
        )
    elif args.vision_model_path is not None and args.language_model_path is not None:
        fusion_model_config = FusionConfig.from_pretrained(args.fusion_model_config_path)
        vision_config = AutoConfig.from_pretrained(args.vision_model_path)
        text_config = AutoConfig.from_pretrained(args.language_model_path)
        vision_text_model_config = VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_config,
            text_config
        )
        oracle_config = OracleConfig.from_vision_text_fusion_configs(
            vision_text_model_config=vision_text_model_config, 
            fusion_model_config=fusion_model_config
        )
    else:
        oracle_config = OracleConfig()

    if args.model_name_or_path is not None:
        model = OracleModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    elif args.vision_language_model_path is not None:
        model = OracleModelForSequenceClassification(oracle_config, vision_language_model_path=args.vision_language_model_path)
    elif args.vision_model_path is not None and args.language_model_path is not None:
        model = OracleModelForSequenceClassification(oracle_config, vision_model_path=args.vision_model_path, language_model_path=args.language_model_path)
    else:
        model = OracleModelForSequenceClassification(oracle_config)

    if args.processor_name is not None:
        processor = AutoProcessor.from_pretrained(args.processor_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path is not None:
        processor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    elif args.vision_model_path is not None and args.language_model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, use_fast=not args.use_slow_tokenizer)
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.vision_model_path)
        processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
    else:
        raise ValueError("You must input the processor_nmae or model_name_or_path or (language_model_path, vision_model_path)")
    
    image_size = vision_text_model_config.vision_config.image_size
    
    def calc_bbox(wh, bbox, size=224):
        
        width, height = wh
        x, y, w, h = bbox
        short, long = (width, height) if width <= height else (height, width)
        requested_new_short = size if isinstance(size, int) else size[0]
        new_short, new_long = requested_new_short, int(requested_new_short * long / short)
        if width <= height:
            nx, nw = x, w
            top = (new_long - size)//2
            y = y * new_long
            h = h * new_long
            ny = min(max(y - top, 0), size)/size
            nh = min(max(h + y - top, 0)/size - ny, size)
        else:
            ny, nh = y, h
            left = (new_long - size)//2
            x = x * new_long
            w = w * new_long

            nx = min(max(x - left, 0), size)/size
            nw = min(max(w + x - left, 0)/size - nx, size)
        
        return (nx, ny, nw, nh)

    calc_bbox_fn = functools.partial(calc_bbox, size=image_size)

    def encode(features):
        features_tmp = processor(text=features["question"], images=[image.convert("RGB") for image in features["image"]], return_tensors="pt")
        wh_l = [image.size for image in features["image"]]
        bbox_l = [bbox for bbox in features["bbox"]]
        
        features_tmp["bbox"] = [calc_bbox_fn(wh, bbox) for wh, bbox in zip(wh_l, bbox_l)]
        
        for k in ["answer", "category"]:
            features_tmp[k] = features[k]
        return features_tmp


    raw_datasets.set_transform(encode)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]

    for index in random.sample(range(len(train_dataset)), 3):
        decoded_inputs = processor.tokenizer.decode(train_dataset[index]["input_ids"])
        shape = train_dataset[index]["pixel_values"].shape
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}. decoded inputs: {decoded_inputs}. Shape: {shape}")

    data_collator = DataCollatorWithPadding(processor.tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))


    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.warmup_portion > 0:
        args.num_warmup_steps = int(args.max_train_steps*max(min(args.warmup_portion, 1), 0))

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Accelerator setting
    device = accelerator.device
    model.to(device)
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader,)
    if args.max_train_steps_per_epoch is not None:
        num_update_steps_per_epoch = min(args.max_train_steps_per_epoch, num_update_steps_per_epoch)
        overrode_max_train_steps = True
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("guesswhat_no_trainer", experiment_config)

    progress_bar = tqdm.auto.tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    starting_epoch = 0
    completed_steps = 0
    metric = evaluate.load("accuracy")

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            for _ in range(num_update_steps_per_epoch*starting_epoch):
                progress_bar.update(1)
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch
            for _ in range(num_update_steps_per_epoch*starting_epoch):
                progress_bar.update(1)


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            """
            input_ids, token_type_ids, attention_mask, pixel_values, answer, category, bbox
            """
            batch["labels"] = batch["answer"]
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1   

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["answer"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                processor.save_pretrained(args.output_dir)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            processor.save_pretrained(args.output_dir)

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    main()
