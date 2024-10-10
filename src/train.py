from peft import LoraConfig

from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import Dataset
from typing import Dict, Optional, List


def main(
    train_dataset_path: str,
    test_dataset_path: str,
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    peft_kwargs: Optional[Dict] = None,
    sft_kwargs: Optional[Dict] = None,
    bnb_config_kwargs: Optional[Dict] = None,
):

    train_dataset = Dataset.load_from_disk(train_dataset_path)
    test_dataset = Dataset.load_from_disk(test_dataset_path)

    # TODO: make this configurable with defaults
    if not peft_kwargs:
        peft_kwargs = dict(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

    # TODO: make this configurable with defaults
    if not bnb_config_kwargs:
        bnb_config_kwargs = dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # TODO: make this configurable with defaults
    if not sft_kwargs:
        sft_kwargs = dict(
            output_dir="qwen2-7b-instruct-amazon-description",
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=5,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,
            # tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            # push_to_hub=True,
            report_to="tensorboard",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )
    # TODO: load the dataset
    peft_config = LoraConfig(**peft_kwargs)
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(**bnb_config_kwargs)

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Make collator a closure so we can access the processor
    def collate_fn(examples: List):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example["messages"], tokenize=False)
            for example in examples
        ]
        image_inputs = [
            process_vision_info(example["messages"])[0] for example in examples
        ]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    args = SFTConfig(**sft_kwargs)
    args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        packing=True,
        data_collator=collate_fn,
        dataset_text_field="",  # needs dummy value
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )
    # start training, the model will be automatically saved to the hub and the output directory
    # if push_to_hub is True
    trainer.train()

    # save model
    trainer.save_model(args.output_dir)
