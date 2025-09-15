"""
train_adapter.py
----------------
Train a small LoRA adapter (PEFT) on toy QA data using distilgpt2 (CPU-friendly).
- Produces: saved adapter folder (./adapters/lora_adapter)
Notes:
- This script is intentionally small for demonstration / CI. Training on CPU is slow but functional.
- Follow SOLID: separated functions for data, model setup, training, and save.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Dict
import torch
from datasets import Dataset
import transformers
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from packaging import version
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

@dataclass
class Config:
    model_name: str = "distilgpt2"
    adapter_name: str = "lora_adapter"
    adapter_dir: str = "./adapters/lora_adapter"
    output_dir: str = "./runs/lora_demo"
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 42
    max_length: int = 128  # truncation / seq length for tokenization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # prefer gpu if present


cfg = Config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_adapter")


# -------------------------
# Toy dataset (few-shot)
# -------------------------
def build_toy_qa() -> List[Dict]:
    # Simple prompt-response pairs (very small)
    samples = [
        {
            "question": "What is a hydraulic pump?",
            "answer": "A hydraulic pump converts mechanical energy into hydraulic energy to move fluid."
        },
        {
            "question": "Why use a charge pump in hydraulic systems?",
            "answer": "To maintain system pressure and prevent cavitation in the main pump at low speeds."
        },
        {
            "question": "What is LoRA used for?",
            "answer": "LoRA is used to fine-tune large models efficiently by adapting low-rank weight updates."
        },
        {
            "question": "What is the unit of pressure?",
            "answer": "Pascal (Pa), equivalent to N/m²; bar and psi are common engineering units too."
        },
    ]
    # Convert to simple text examples for causal LM training: "<Q> {question}\n<A> {answer}"
    examples = []
    for i, s in enumerate(samples):
        prompt = f"Q: {s['question']}\nA:"
        # The target for generation should contain the answer plus an end token; we keep simple.
        text = prompt + " " + s["answer"]
        examples.append({"id": f"t{i}", "text": text})
    return examples


# -------------------------
# Tokenization & dataset
# -------------------------
def prepare_dataset(tokenizer: PreTrainedTokenizer, examples: List[Dict], max_length: int):
    texts = [e["text"] for e in examples]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    ds = Dataset.from_dict({k: v.tolist() for k, v in enc.items()})
    return ds


# -------------------------
# Model + LoRA setup
# -------------------------
def build_model_and_peft(tokenizer: PreTrainedTokenizer):
    logger.info("Loading base model: %s", cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    # Ensure pad token exists (GPT2 family often don't set it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for PEFT (if using int8 or kbit training; here minimal)
    # For CPU-only, keep default (no 8bit). Still call prepare_model_for_kbit_training for correctness if needed.
    try:
        model = prepare_model_for_kbit_training(model)
    except (AttributeError, ImportError) as e:
        # Not fatal on CPU / small models; proceed with vanilla model
        logger.debug("prepare_model_for_kbit_training skipped: %s", e)

    # LoRA config - small ranks for tiny demo
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # DistilGPT2 style: target c_attn; may vary by model
    )
    model = get_peft_model(model, lora_config)
    logger.info("Wrapped model with LoRA PEFT.")
    return model


# -------------------------
# Train
# -------------------------
def train():
    torch.manual_seed(cfg.seed)
    examples = build_toy_qa()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = prepare_dataset(tokenizer, examples, max_length=cfg.max_length)

    # split tiny dataset into train/eval (80/20)
    split = dataset.train_test_split(test_size=0.25, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    model = build_model_and_peft(tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    TRANSFORMERS_VERSION = version.parse(transformers.__version__)

    base_args = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=False,
        seed=cfg.seed,
    )

    # Newer Transformers (>=4.20) → use evaluation_strategy/save_strategy
    if TRANSFORMERS_VERSION >= version.parse("4.20.0"):
        training_args = TrainingArguments(
            **base_args,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            push_to_hub=False,
            report_to=[],
        )

    # Older Transformers (between 3.0 and 4.19) → no evaluation_strategy, use eval_steps
    elif TRANSFORMERS_VERSION >= version.parse("3.0.0"):
        training_args = TrainingArguments(
            **base_args,
            eval_steps=500,  # run evaluation every N steps
            save_steps=500,
            logging_dir="./logs",
        )

    # Very old (<3.0) fallback
    else:
        training_args = TrainingArguments(
            **base_args,
            save_steps=500,
            logging_dir="./logs",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    logger.info("Starting training on device: %s", cfg.device)
    trainer.train()
    # Save only adapter weights (PEFT adapter)
    os.makedirs(cfg.adapter_dir, exist_ok=True)
    logger.info("Saving LoRA adapter to %s", cfg.adapter_dir)
    model.save_pretrained(cfg.adapter_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    train()