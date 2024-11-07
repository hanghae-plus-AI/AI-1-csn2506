from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

def formatting_prompts_func(examples):
    formatted_input = []
    for instruction, input_text in zip(examples["instruction"], examples["input"]):
        if input_text:
            formatted_input.append(f"{instruction}\n\n{input_text}")
        else:
            formatted_input.append(instruction)
    
    model_inputs = tokenizer(
        formatted_input, 
        truncation=True, 
        padding="max_length",
        max_length=128,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

formatted_dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

collator = None

lora_r_values = [8, 128, 256]
for lora_r in lora_r_values:
    print(f"LoRA rank: {lora_r}로 학습 시작")
    lora_config = LoraConfig(r=lora_r)
    lora_model = get_peft_model(model, lora_config)
    sft_config = SFTConfig(
        output_dir=f"/tmp/clm-instruction-tuning-lora_r_{lora_r}",
        max_seq_length=128
    )
    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=formatted_dataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.train()