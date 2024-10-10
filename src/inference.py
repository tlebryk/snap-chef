import argparse
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from datasets import Dataset


# TODO: doesn't support batch inference properly right now
def generate_qwen_outputs(messages, model, processor, generation_params=None):
    # set some default generation params
    if not generation_params:
        generation_params = {
            "max_new_tokens": 256,
            "top_p": 1.0,
            "do_sample": True,
            "temperature": 0.8,
        }
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main(
    input_dataset_path: str,
    output_dataset_file: str,
    adapter_path=None,
    bnb_config_kwargs=None,
):
    if not bnb_config_kwargs:
        bnb_config_kwargs = dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    dataset = Dataset.load_from_disk(input_dataset_path)
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        **bnb_config_kwargs,
    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    if adapter_path:
        model.load_adapter(adapter_path)
    processor = AutoProcessor.from_pretrained(model_id)
    # save predictions and evaluate against true answer in 'answer' column of dataset
    for i in range(len(dataset)):
        # generate qwen outputs
        qwen_outputs = generate_qwen_outputs(dataset[i]["messages"], model, processor)
        print(qwen_outputs)
        # save predictions
        dataset[i]["predictions"] = qwen_outputs

    # save just the predictions and index
    dataset = dataset.map(
        lambda x: {"predictions": x["predictions"], "index": x["index"]}
    )
    dataset.save_to_disk(output_dataset_file)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions for a dataset using Qwen2-VL-7B-Instruct"
    )
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        help="Path to the dataset to generate predictions for",
    )
    parser.add_argument(
        "--output_dataset_file",
        type=str,
        help="Path to save the dataset with predictions",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Path to the adapter to load",
    )
    args = parser.parse_args()
    main(
        args.input_dataset_path,
        args.output_dataset_file,
        args.adapter_path,
    )
