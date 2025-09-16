import json
import gc
import torch
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Login with HF key
hf_key = os.getenv("HFKEYT")
login(hf_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data")
json_FILE = os.path.join(SAVE_DIR, 'evaluation.json')

# Load test questions
with open(json_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test questions")
for i, item in enumerate(test_data[:3], 1):
    print(f"Q{i}: {item['question'][:100]}...")

# -------------------------------
# Load Base Model
# -------------------------------
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
print(f"Loading base model: {base_model_name}")

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,   # or torch.float16
    load_in_4bit=True             # requires bitsandbytes
)

print("Base model loaded successfully!")

# Inference with base model
system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."
base_answers = []

print("Testing base model...")
for i, item in enumerate(test_data, 1):
    question = item['question']
    prompt = f"<|system|>{system_prompt}<|user|>{question}<|assistant|>"

    inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.1,
            pad_token_id=base_tokenizer.eos_token_id
        )

    full_response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split('<|assistant|>')[-1].strip()

    base_answers.append(answer)
    print(f"Q{i} completed")

print(f"Base model testing completed. Generated {len(base_answers)} answers.")

# Display base answers
print("=== BASE MODEL ANSWERS ===")
for i, (item, answer) in enumerate(zip(test_data, base_answers), 1):
    print(f"\nQ{i}: {item['question']}")
    print(f"Base Answer: {answer}")
    print("-" * 80)

# Free GPU memory
del base_model
del base_tokenizer
torch.cuda.empty_cache()
gc.collect()
print("Base model cleared from memory")

# -------------------------------
# Load Fine-Tuned Model (with LoRA adapters)
# -------------------------------
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
ft_adapter_path = "llama3.1-7b-qa-finetuned"  # path or HF repo id
print(f"Loading base model: {base_model_name}")
print(f"Loading LoRA adapters from: {ft_adapter_path}")

try:
    ft_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    ft_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    ft_model = PeftModel.from_pretrained(ft_base_model, ft_adapter_path)
    print("LoRA adapters loaded successfully! Fine-tuned model ready.")

except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    ft_model, ft_tokenizer = None, None

# -------------------------------
# Inference with Fine-Tuned Model
# -------------------------------
ft_answers = []
if ft_model is not None:
    print("Testing fine-tuned model...")
    for i, item in enumerate(test_data, 1):
        question = item['question']
        prompt = f"<|system|>{system_prompt}<|user|>{question}<|assistant|>"

        inputs = ft_tokenizer(prompt, return_tensors="pt").to(ft_model.device)

        with torch.no_grad():
            outputs = ft_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.1,
                pad_token_id=ft_tokenizer.eos_token_id
            )

        full_response = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split('<|assistant|>')[-1].strip()

        ft_answers.append(answer)
        print(f"Q{i} completed")

    print(f"Fine-tuned model testing completed. Generated {len(ft_answers)} answers.")
else:
    ft_answers = ["Model not available"] * len(test_data)

# Display fine-tuned answers
if ft_model is not None:
    print("=== FINE-TUNED MODEL ANSWERS ===")
    for i, (item, answer) in enumerate(zip(test_data, ft_answers), 1):
        print(f"\nQ{i}: {item['question']}")
        print(f"Fine-tuned Answer: {answer}")
        print("-" * 80)

# Free GPU memory
if ft_model is not None:
    del ft_model
    if 'ft_base_model' in locals():
        del ft_base_model
    del ft_tokenizer
torch.cuda.empty_cache()
gc.collect()
print("Fine-tuned model cleared from memory")

# -------------------------------
# Save Comparison Results
# -------------------------------
print("=== MODEL COMPARISON ANALYSIS ===")
comparison_results = []
for i, (item, base_ans, ft_ans) in enumerate(zip(test_data, base_answers, ft_answers), 1):
    question = item['question']
    expected = item['answer']

    print(f"\nQUESTION {i}: {question}")
    print(f"EXPECTED: {expected}")
    print(f"BASE: {base_ans}")
    print(f"FINE-TUNED: {ft_ans}")

    comparison_results.append({
        "question_id": i,
        "question": question,
        "expected_answer": expected,
        "base_answer": base_ans,
        "finetuned_answer": ft_ans,
    })

results = os.path.join(SAVE_DIR, 'results.json')
with open(results, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)
