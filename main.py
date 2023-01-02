from base64 import encode
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedDataParallelKwargs
from transformers.utils import logging as transformers_logging
# from transformers import CLIPProcessor
from transformers import (
    AutoProcessor,
    SchedulerType,
    get_scheduler,
)
from datasets.utils import logging as datasets_logging
from models.oracle import OracleModel
from torch.optim import AdamW
from datetime import timedelta
import argparse, math, tqdm, os, random, logging, torch

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--weight_deacy", type=float, default=0.0,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=8e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
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
        "--gradient_accumulation_steps",type=int, default=1,
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
    with accelerator.main_process_first():
        raw_datasets = load_dataset(
            "guesswhat.py",
            "base",
            cache_dir="huggingface_datasets",
            data_dir="data",
            ignore_verifications=True,
        )
    train_dataloader = raw_datasets["train"]
    eval_dataloader = raw_datasets["validation"]
    test_dataloader = raw_datasets["test"]

    # Config Setting
    model = OracleModel.from_pretrained("oracle")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
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
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps_per_epoch is not None:
        num_update_steps_per_epoch = min(args.max_train_steps_per_epoch, num_update_steps_per_epoch)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_tokenizer
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Accelerator setting
    device = accelerator.device
    model.to(device)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps_per_epoch is not None:
        num_update_steps_per_epoch = min(args.max_train_steps_per_epoch, num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    starting_epoch = 0
    completed_steps = 0

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
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if (resume_step is not None
                    and step < resume_step
                    and step % args.gradient_accumulation_steps == 0):
                    completed_steps += 1
                    progress_bar.update(1)
                    continue
            if args.max_train_steps_per_epoch is not None and (step//args.gradient_accumulation_steps) >= num_update_steps_per_epoch:
                break

            inputs = processor(text=batch["qas"]["q"], images=batch["image"], return_tensors="pt", padding=True)
            logits = processor(text=batch["qas"]["a"])
            # various sizes to (336, 336)
            for i in range(batch):
                temp_mask = torch.unsqueeze(bbox2_mask(batch[i]), 0)
                if i == 0:
                    attention_mask = temp_mask
                else:
                    attention_mask = torch.cat((attention_mask, temp_mask), 0)

            outputs = model(
                **inputs,
                attention_mask=attention_mask,
            )
            loss = logits, outputs.hidden_states
            accelerator.backward(loss)

            
            if args.with_tracking:
                total_loss += loss.detach().float()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                progress_bar.update(1)
                completed_steps += 1
            
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and step % args.gradient_accumulation_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if (args.logging_steps>0 and completed_steps % args.logging_steps == 0 and completed_steps > 0) and step % args.gradient_accumulation_steps == 0:
                logger.info(
                    "train_loss: {:.3f}".format(
                        total_loss.item()/step, 
                        )
                    )

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        total_eval_loss = 0

        for step, batch in enumerate(eval_dataloader):
            if args.max_validation_steps is not None and step >= args.max_validation_steps:
                break
            
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    attention_mask=attention_mask,
                )
                loss = logits, outputs.hidden_states
                loss = args.captioning_weight*loss
                total_eval_loss += accelerator.reduce(loss).detach().float()

        logger.info("Evaluation - loss: {}, mrr: {}, lm_loss".format(
                total_eval_loss.item() / accelerator.num_processes / len(eval_dataloader),
            ))
        
        result = {}
        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["eval_loss"] = total_eval_loss.item() / accelerator.num_processes / len(eval_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)
        
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            image_tokenizer.save_pretrained(args.output_dir)
        
        if result is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(
                    {
                        "train_loss": result["train_loss"],
                        "train_mrr": result["train_mrr"],
                        "train_lm_loss": result["train_lm_loss"],
                        "eval_loss": result["eval_loss"],
                        "eval_mrr": result["eval_mrr"],
                        "total_eval_lm_loss": result["total_eval_lm_loss"],
                    },
                    f,
                )
if __name__ == "__main__":
    main()


def bbox_to_attention_mask(batch):
    answer_id = batch["object_id"]
    bbox = [0., 0., 0., 0.]
    for k in range(batch["objects"]):
        if k["object_id"] == answer_id:
            bbox = k["bbox"]
        else:
            pass
    
    return bbox2_mask(bbox, batch["height"], batch["width"])

def bbox2_mask(bbox, img_H, img_W):  # [B, head, query, key]   --> [B, head, inp_seq_len, 1]
    x1, y1, x2, y2 = bbox[:,:,0], bbox[:,:,1], bbox[:,:,2]+bbox[:,:,0], bbox[:,:,3]+bbox[:,:,1]  # [B, 1]
    h_linspace = torch.unsqueeze(torch.linspace(0, img_H-1, steps=img_H), 0) #  [1, H]
    w_linspace = torch.unsqueeze(torch.linspace(0, img_W-1, steps=img_W), 0)  # [1, W]
    # print(w_linspace.shape, torch.tile(x1, (1, img_W)).shape)
    x1_bool = torch.le(torch.tile(x1, (1, img_W)), w_linspace)  # [1, W] [B, W]
    x2_bool = torch.le(w_linspace, torch.tile(x2, (1, img_W)))  # [1, W] [B, W]
    y1_bool = torch.le(torch.tile(y1, (1, img_H)), h_linspace)  # [1, H] [B, H]
    y2_bool = torch.le(h_linspace, torch.tile(y2, (1, img_H)))  # [1, H] [B, H]
    x_bool = torch.unsqueeze(x1_bool * x2_bool, -2)  # [B, 1, W]
    y_bool = torch.unsqueeze(y1_bool * y2_bool, -1)  # [B, H, 1]
    x_map = torch.tile(x_bool, (1, img_H, 1))  # [B, H, W]
    y_map = torch.tile(y_bool, (1, 1, img_W))  # [B, H, W]
    mask = torch.unsqueeze((x_map * y_map).to(torch.float32), 1)
    return mask

"""
# tokenizer, processor setting
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    # encoder model setting
    encoder_model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        "google/vit-base-patch16-224", "bert-base-uncased"
    )

    # oracle model setting
    oracle_model = FusionModel.from_pretrained("")

    # accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # model to accel
    encoder_model.to(device)
    oracle_model.to(device)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    images = []
    inputs = processor(

    )
    outputs = encoder_model(
        input_ids = inputs.input_ids,
        attentiom_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_Values,
        return_loss=True,
    )
    
    text_model_output=outputs.text_outputs,
    vision_model_output=outputs.vision_outputs,
    outputs = oracle_model(
        input_ids = torch.cat((vision_model_output, text_model_output), 1)
    )
    # loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score

"""