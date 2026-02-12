# Instructions: Fine-Tuning an Open-Source LLM with QLoRA

Follow the steps below to complete the assignment.  
You are expected to understand _why_ each step exists, not just run code.

## Step 1: Environment Setup

You may work **locally with a GPU** or use **Google Colab**.

### Option A: Google Colab (Recommended)

1. Open a new Colab notebook
2. Runtime → Change runtime type → **GPU**
3. Install dependencies:

```bash
pip install -U \
  torch transformers datasets accelerate \
  bitsandbytes peft trl \
  huggingface-hub python-dotenv \
  jupyter ipython notebook
```

### Option B: Local Environment (Python 3.10+)

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

(Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -U \
  torch transformers datasets accelerate \
  bitsandbytes peft trl \
  huggingface-hub python-dotenv \
  jupyter ipython notebook
```

> Ensure PyTorch is installed with CUDA support if running locally.

3. Log in to Hugging Face if you don't have already:

```bash
huggingface-cli login
```

## Step 2: Dataset Exploration & Preparation

### Dataset

You will use the **Bitext Customer Support Dataset** from Hugging Face:

[https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

### Goals

- Understand the dataset structure
- Identify relevant fields (`instruction`, `response`)
- Convert examples into a **chat-style instruction format**

### Example Exploration

```python
from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

dataset["train"][0]
```

### Formatting Strategy

You should format each example as a **single-turn conversation**:

- System: defines the assistant’s role
- User: customer query
- Assistant: support agent response

Example format (conceptual):

```python
[
  {"role": "system", "content": "You are a helpful customer support assistant."},
  {"role": "user", "content": instruction},
  {"role": "assistant", "content": response}
]
```

You may perform this transformation **in memory** using `datasets.map()` or inside your training script.

> There is no required on-disk format. Choose what best fits your workflow.

## Step 3: Train the Model with QLoRA

### Model (training only)

- **Base Model**: Llama 3.2 3B (from Hugging Face; used only for training)
- **Method**: QLoRA (4-bit quantization + LoRA adapters)

> **Note:** Training always uses the Hugging Face model above. For **inference** (evaluate & demo) you can instead use a **local Ollama model** (e.g. Gemma 12B) with `--ollama`; see “Inference options” at the end.

### Requirements

Your training code must:

- Load the base model in **4-bit**
- Attach LoRA adapters using `peft`
- Use either:
  - `trl.SFTTrainer`, or
  - Hugging Face `Trainer`

- Train only adapter weights (not full fine-tuning)

Recommended starting settings:

- Sequence length: 512–1024
- Batch size: small (1–4)
- Gradient accumulation: enabled
- Epochs: 1
- Seed: 42

> Tip: First test training on a **small subset** to ensure everything works.

## Step 4: Evaluation

### Objective

Demonstrate model behavior on customer support prompts (either compare base vs fine-tuned, or run a single model such as Ollama).

### Required Evaluation

1. Create **10–15 custom test prompts**
   - Must not be copied from the dataset
   - Should reflect real customer support scenarios

2. **Option A – Hugging Face:** Compare base vs fine-tuned Llama:
   - Base model responses
   - Fine-tuned model responses
   - Save a side-by-side comparison (prompt, base output, fine-tuned output, commentary).

3. **Option B – Ollama (localhost):** Run prompts through your local Ollama model (e.g. Gemma 12B):
   - `python main.py evaluate --ollama --ollama-model gemma3:12b --output ollama_eval.json`
   - Results contain prompt + single model output (no base vs fine-tuned comparison).

Focus on:

- Helpfulness
- Tone and professionalism
- Specificity (less generic responses)

## Step 5: Demo (Optional but Encouraged)

Create a simple demo using **Gradio** that:

- **Option A – Hugging Face:** Loads the base model, applies your fine-tuned adapter, accepts user input, displays the model’s response.
- **Option B – Ollama (localhost):** Uses your local Ollama model (no Llama/adapter load):
  - `python main.py demo --ollama --ollama-model gemma3:12b`
  - Full chat history is sent each turn so multi-turn answers stay correct.

This demonstrates real-world usability.

---

## Inference options (no Llama required)

If you prefer **not** to use Llama 3.2 for inference (e.g. you use a local Ollama model instead):

| Step    | With Llama / Hugging Face              | With Ollama (localhost)                    |
|---------|----------------------------------------|-------------------------------------------|
| **Train**   | Llama 3.2 3B + QLoRA → adapter saved   | *(Training still uses Llama; Ollama is inference-only.)* |
| **Evaluate**| Base vs fine-tuned comparison          | `evaluate --ollama --ollama-model gemma3:12b` → single model outputs |
| **Demo**    | Load adapter, chat in Gradio           | `demo --ollama` → Gradio uses Ollama (e.g. Gemma 12B), full chat history |

Ensure Ollama is running (`ollama serve`) and the model is pulled (e.g. `ollama pull gemma3:12b`).
