#!/usr/bin/env python3
"""
Customer Support LLM Fine-Tuning with QLoRA

Fine-tunes Llama 3.2 3B on the Bitext Customer Support dataset using QLoRA.
Implements: data preparation, training, evaluation, and Gradio demo.

Usage:
    python main.py train [--subset N] [--output_dir PATH]
    python main.py evaluate [--adapter_path PATH]
    python main.py demo [--adapter_path PATH]
"""

import argparse
import inspect
import json
import os
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# =============================================================================
# Configuration
# =============================================================================

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
SYSTEM_PROMPT = "You are a helpful customer support assistant."

# Training defaults (from PLAN.md, Instructions.md, customer-support-plan)
TRAINING_CONFIG = {
    "max_seq_length": 1024,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "epochs": 1,
    "learning_rate": 2e-4,
    "seed": 42,
    "test_size": 0.05,
}

# 10-15 custom test prompts (NOT from dataset) - from evaluation requirements
TEST_PROMPTS = [
    "I need to cancel my order #12345 as soon as possible.",
    "Where is my package? I ordered it 5 days ago.",
    "How do I get a refund for a defective product?",
    "I want to change the shipping address for my order.",
    "My payment was charged twice, please help.",
    "Can I return an item after 30 days?",
    "I need to cancle my order",  # Typo - robustness test
    "Order 98765 please cancel it",  # Structural variation
    "What's the status of my refund request?",
    "How do I track my shipment?",
    "I need help with my account login.",
    "Cancel everything",  # Edge case - ambiguity
    "What's the weather today?",  # Out-of-scope
    "I wanna get my money back for order X",  # Paraphrase
    "Help me with a return - the item doesn't fit.",
]

# Llama 3.2 chat template (base model has none; SFTTrainer needs it)
LLAMA_32_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)

# =============================================================================
# Data Preparation
# =============================================================================


def _set_chat_template_if_missing(tokenizer):
    """Set Llama 3.2 chat template on tokenizer when missing (e.g. base model)."""
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = LLAMA_32_CHAT_TEMPLATE


def format_as_chat(example: dict) -> dict:
    """Convert dataset example to chat format for SFT."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def prepare_dataset(
    subset_size: int | None = None,
    seed: int = 42,
    test_size: float = 0.05,
) -> tuple:
    """
    Load Bitext dataset, format as chat, create stratified train/val split.

    Returns:
        (train_dataset, eval_dataset)
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(DATASET_NAME)

    # Get train split (dataset has only 'train')
    ds = dataset["train"]

    # train_test_split: stratify_by_column requires ClassLabel, Bitext has
    # intent as string, so use regular split
    split = ds.train_test_split(test_size=test_size, seed=seed)

    # Format as chat (system, user, assistant)
    def format_and_remove(ex):
        formatted = format_as_chat(ex)
        return formatted

    # Map - remove original columns, keep only messages
    original_columns = split["train"].column_names
    split = split.map(
        format_and_remove,
        remove_columns=original_columns,
        desc="Formatting as chat",
    )

    train_ds = split["train"]
    eval_ds = split["test"]

    # Optional: subset for quick testing
    if subset_size is not None:
        train_ds = train_ds.select(range(min(subset_size, len(train_ds))))
        eval_size = min(50, len(eval_ds))
        eval_ds = eval_ds.select(range(eval_size))
        print(f"Using subset: {len(train_ds)} train, {len(eval_ds)} eval")

    print(f"Train examples: {len(train_ds)}, Eval examples: {len(eval_ds)}")
    return train_ds, eval_ds


# =============================================================================
# Training
# =============================================================================


def get_model_and_tokenizer():
    """Load model in 4-bit (QLoRA) when possible, else bfloat16 + LoRA on Mac."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _set_chat_template_if_missing(tokenizer)

    used_4bit = True
    device_map = None
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        print("Using QLoRA (4-bit quantization + LoRA)")
    except (ImportError, OSError) as e:
        err_msg = str(e).lower()
        if "cuda" in err_msg or "bitsandbytes" in err_msg:
            used_4bit = False
            # Explicit device avoids disk offload (mps for Apple Silicon, else cpu)
            try:
                device_map = "mps" if torch.backends.mps.is_available() else "cpu"
            except AttributeError:
                device_map = "cpu"
            # MPS does not support bfloat16; use float16 for Apple Silicon
            dtype = torch.float16 if device_map == "mps" else torch.bfloat16
            print(
                "CUDA/bitsandbytes unavailable (e.g. Mac). "
                f"Using {dtype} + LoRA on {device_map}."
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            raise

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # When fallback on MPS we use float16; training must use fp16 not bf16
    use_fp16 = device_map == "mps"
    return model, tokenizer, used_4bit, use_fp16


def train(
    output_dir: str = "output/customer-support-llm",
    subset_size: int | None = None,
):
    """Run QLoRA fine-tuning."""
    train_ds, eval_ds = prepare_dataset(
        subset_size=subset_size,
        seed=TRAINING_CONFIG["seed"],
        test_size=TRAINING_CONFIG["test_size"],
    )

    model, tokenizer, used_4bit, use_fp16 = get_model_and_tokenizer()

    # fp16 + MPS requires PyTorch >= 2.5 (accelerate limitation)
    mps_used = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    if use_fp16 and mps_used:
        try:
            parts = torch.__version__.split("+")[0].split(".")
            major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            if (major, minor) < (2, 5):
                use_fp16 = False
                print(
                    "PyTorch < 2.5: disabling fp16 on MPS (using float32)."
                )
        except (ValueError, IndexError):
            use_fp16 = False

    # MPS does not support bf16; use bf16 only on CUDA/4-bit path
    use_bf16 = (not use_fp16) and (not mps_used)

    # Use smaller batch when not using 4-bit (e.g. Mac) to avoid OOM
    batch_size = TRAINING_CONFIG["batch_size"] if used_4bit else 1
    grad_accum = TRAINING_CONFIG["gradient_accumulation_steps"]
    if not used_4bit:
        grad_accum = min(16, grad_accum * 2)  # Keep effective batch size similar

    # SFTConfig (trl) holds both training args and max_length
    max_len = TRAINING_CONFIG["max_seq_length"]
    try:
        sft_args = SFTConfig(
            output_dir=output_dir,
            max_length=max_len,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=TRAINING_CONFIG["epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            seed=TRAINING_CONFIG["seed"],
            report_to="none",
            packing=False,
        )
    except TypeError:
        # Older trl: use TrainingArguments and pass max_seq_length to SFTTrainer
        sft_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=TRAINING_CONFIG["epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            seed=TRAINING_CONFIG["seed"],
            report_to="none",
        )

    # SFTTrainer: max_length in SFTConfig; tokenizer as processing_class (trl)
    trainer_kw = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    # Older trl: SFTTrainer takes max_seq_length and tokenizer=
    if "max_seq_length" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kw["max_seq_length"] = max_len
        trainer_kw["packing"] = False
        trainer_kw.pop("processing_class", None)
        trainer_kw["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kw)

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and adapter saved to {output_dir}")
    return output_dir


# =============================================================================
# Inference Helpers
# =============================================================================


def _load_base_model():
    """Load base model; 4-bit if CUDA/bitsandbytes available, else bfloat16."""
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except (ImportError, OSError) as e:
        if "cuda" in str(e).lower() or "bitsandbytes" in str(e).lower():
            try:
                device_map = "mps" if torch.backends.mps.is_available() else "cpu"
            except AttributeError:
                device_map = "cpu"
            dtype = torch.float16 if device_map == "mps" else torch.bfloat16
            return AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        raise


def load_model_for_inference(adapter_path: str | None = None, base_only: bool = False):
    """Load model (base or base + adapter) for inference."""
    if adapter_path and not base_only:
        model = _load_base_model()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        model = _load_base_model()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _set_chat_template_if_missing(tokenizer)

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    user_message: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate assistant response given user message."""
    # Chat template for Llama 3.2
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (assistant response)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant reply (after the user message)
    is_assistant = (
        "assistant" in full_output.lower() or
        "<|start_header_id|>assistant" in full_output
    )
    if is_assistant:
        parts = full_output.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            # Clean up any trailing tokens
            if "<|" in response:
                response = response.split("<|")[0].strip()
            return response
    return full_output


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(
    adapter_path: str | None = None,
    output_path: str = "evaluation_results.json",
):
    """
    Compare base vs fine-tuned model on 10-15 custom test prompts.
    Saves side-by-side comparison with commentary.
    """
    print("Loading base model...")
    base_model, base_tokenizer = load_model_for_inference(base_only=True)

    print("Loading fine-tuned model...")
    if adapter_path and os.path.exists(adapter_path):
        ft_model, ft_tokenizer = load_model_for_inference(adapter_path)
    else:
        print(
            "No adapter path or path missing. Using base model for both."
        )
        ft_model, ft_tokenizer = base_model, base_tokenizer

    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"Evaluating {i + 1}/{len(TEST_PROMPTS)}: {short}")

        base_response = generate_response(base_model, base_tokenizer, prompt)
        ft_response = generate_response(ft_model, ft_tokenizer, prompt)

        # Simple heuristic commentary
        base_len = len(base_response)
        ft_len = len(ft_response)
        commentary = (
            f"Base: {base_len} chars. Fine-tuned: {ft_len} chars. "
            "Fine-tuned should be more specific and professional."
        )

        results.append({
            "prompt": prompt,
            "base_output": base_response,
            "fine_tuned_output": ft_response,
            "commentary": commentary,
        })

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (first 3 prompts)")
    print("=" * 80)
    for r in results[:3]:
        print(f"\nPrompt: {r['prompt']}")
        print(f"Base:      {r['base_output'][:200]}...")
        print(f"Fine-tuned: {r['fine_tuned_output'][:200]}...")
        print("-" * 40)

    return results


# =============================================================================
# Gradio Demo
# =============================================================================


def run_demo(adapter_path: str | None = None):
    """Launch Gradio demo for customer support chatbot."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return

    # Suppress harmless console warnings during demo
    warnings.filterwarnings(
        "ignore",
        message="To copy construct from a tensor",
        category=UserWarning,
        module="transformers",
    )
    warnings.filterwarnings(
        "ignore",
        message="bitsandbytes was compiled without GPU",
        category=UserWarning,
        module="bitsandbytes",
    )
    warnings.filterwarnings(
        "ignore",
        message="torch_dtype.*deprecated",
        category=UserWarning,
        module="transformers",
    )

    print("Loading model (this may take a minute)...")
    model, tokenizer = load_model_for_inference(adapter_path)

    # Shorter default max tokens for faster replies on CPU/MPS (each token = 1 forward pass)
    DEFAULT_MAX_TOKENS = 192

    with gr.Blocks(title="Customer Support Chatbot") as demo:
        gr.Markdown("# Customer Support Chatbot (Fine-tuned Llama 3.2 3B)")
        gr.Markdown(
            "Ask about orders, shipping, refunds, returns, and more. "
            "Lower **Max response tokens** for faster replies (e.g. on Mac)."
        )

        chatbot = gr.Chatbot(label="Conversation", type="messages")
        max_tokens_slider = gr.Slider(
            minimum=64,
            maximum=512,
            value=DEFAULT_MAX_TOKENS,
            step=32,
            label="Max response tokens (lower = faster)",
        )
        msg = gr.Textbox(
            label="Your question",
            placeholder="e.g., I need to cancel my order #12345",
            lines=2,
        )

        with gr.Row():
            submit = gr.Button("Send")
            clear = gr.Button("Clear")

        def respond(message, chat_history, max_tokens):
            if not message.strip():
                return chat_history, ""
            n = int(max_tokens) if max_tokens else DEFAULT_MAX_TOKENS
            response = generate_response(
                model, tokenizer, message, max_new_tokens=n
            )
            new_messages = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return new_messages, ""

        inputs = [msg, chatbot, max_tokens_slider]
        msg.submit(respond, inputs, [chatbot, msg])
        submit.click(respond, inputs, [chatbot, msg])
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(share=False)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Customer Support LLM Fine-Tuning with QLoRA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser(
        "train", help="Run QLoRA fine-tuning"
    )
    train_parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Train on N examples (testing). Omit for full training.",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/customer-support-llm",
        help="Output directory for model and adapter",
    )

    # Evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Compare base vs fine-tuned"
    )
    eval_parser.add_argument(
        "--adapter_path",
        type=str,
        default="output/customer-support-llm",
        help="Path to fine-tuned adapter",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Launch Gradio demo")
    demo_parser.add_argument(
        "--adapter_path",
        type=str,
        default="output/customer-support-llm",
        help="Adapter path (omit for base model)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            output_dir=args.output_dir,
            subset_size=args.subset,
        )
    elif args.command == "evaluate":
        evaluate(
            adapter_path=args.adapter_path,
            output_path=args.output,
        )
    elif args.command == "demo":
        run_demo(adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
