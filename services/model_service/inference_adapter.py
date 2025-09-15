"""
inference_adapter.py
--------------------
Load base model + LoRA adapter and generate answers for QA prompts.

Usage:
    python inference_adapter.py --adapter_path ./adapters/lora_adapter --prompt "Q: What is LoRA?\nA:"

This script keeps the code compact and clear: model loading separated from generation.
"""

import argparse
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("inference_adapter")
logging.basicConfig(level=logging.INFO)


def load_model_and_adapter(base_model: str, adapter_path: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model (%s) in eval mode.", base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.eval()
    # Load PEFT adapter weights by calling from_pretrained on the model object if adapter present.
    if adapter_path:
        try:
            # PEFT adapters saved with save_pretrained can be loaded via `from_pretrained` into the model
            # Since we didn't wrap model with get_peft_model in this script, use PEFT's recommended way:
            from peft import PeftModel

            logger.info("Loading adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as exc:
            logger.warning("Failed to load adapter via PeftModel: %s", exc)
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 64):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return only the generated continuing text (strip input prompt)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="./adapters/lora_adapter", help="Path to saved adapter")
    parser.add_argument("--base_model", type=str, default="distilgpt2", help="Base model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation (e.g., 'Q: ...\\nA:')")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    tokenizer, model = load_model_and_adapter(args.base_model, args.adapter_path)
    ans = generate_answer(tokenizer, model, args.prompt, max_new_tokens=args.max_new_tokens)
    print("=== GENERATED ===")
    print(ans)


if __name__ == "__main__":
    main()
