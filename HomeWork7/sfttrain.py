import json
from pathlib import Path
from datasets import load_dataset
import torch
import warnings
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import os
from huggingface_hub import login


hf_key=os.getenv("HFKEYT")
login(hf_key)

# ---------- Step 1: Convert qa.json into JSONL format ----------
system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data")                             # Number of papers to fetch
json_FILE = os.path.join(SAVE_DIR,"qa.json")

with open(json_FILE, "r") as infile:
    qas_list = json.load(infile)

data = []
for qa in qas_list:
    user_q = qa["question"]
    assistant_a = qa["answer"]
    full_prompt = f"<|system|>{system_prompt}<|user|>{user_q}<|assistant|>{assistant_a}"
    data.append({"text": full_prompt})

jsonl_path = Path("qa_sft.jsonl")
with open(jsonl_path, "w") as outfile:
    for entry in data:
        outfile.write(json.dumps(entry) + "\n")

print(f"✅ Converted qa.json -> {jsonl_path}")


# ---------- Step 2: Fine-tuning Function ----------
def fine_tune_llama_model(
    # model_name: str = "unsloth/llama-3.1-7b-unsloth-bnb-4bit",
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    dataset_file: str = "qa_sft.jsonl",
    output_dir: str = "llama3.1-7b-qa-finetuned",
    batch_size: int = 4,
    gradient_steps: int = 4,
    epochs: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048
):
    """
    Fine-tune LLaMA 3.1 7B model using Hugging Face + PEFT LoRA
    """


    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Prevent pad errors

    model_options = [
    "meta-llama/Llama-3.2-1B-Instruct",
     ]
    
    for model_name in model_options:
        try:
            print(f"🔄 Trying {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            # functioncall LoRA configuration
            if "llama" in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Focus on attention
                r, alpha = 32, 64  # Moderate values
            elif "phi" in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj"]
                r, alpha = 16, 32
            elif "gemma" in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj"]
                r, alpha = 16, 32
            else:
                target_modules = ["c_attn", "c_proj"]
                r, alpha = 32, 64
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=r,  # functioncall rank
                lora_alpha=alpha,  # functioncall alpha
                lora_dropout=0.05,  # Small dropout for regularization
                target_modules=target_modules,
                bias="none"
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"❌ Failed {model_name}: {e}")
            continue
    print(f"📂 Loading dataset: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"📊 Dataset size: {len(dataset)} samples")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=384,
        )
        result["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids]
            for ids in result["input_ids"]
        ]
        return result

    # dataset = Dataset.from_list(dataset)
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],
        batched=True
    )


    training_args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Moderate epochs
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=10,  # Small warmup
        learning_rate=3e-4,  # Moderate learning rate
        fp16=False,
        logging_steps=2,
        save_steps=20,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,
        dataloader_num_workers=0,
        weight_decay=0.01,  # Small weight decay for regularization
        lr_scheduler_type="cosine",  # Gradual decay
        max_grad_norm=1.0,  # Gradient clipping
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("🔥 Starting fine-tuning...")
    trainer.train()

    print(f"💾 Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Fine-tuning completed!")

    return model, tokenizer


if __name__ == "__main__":
    fine_tune_llama_model()
