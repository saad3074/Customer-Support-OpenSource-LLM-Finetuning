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

### Model

- **Base Model**: Llama 3.2 3B
- **Method**: QLoRA (4-bit quantization + LoRA adapters)

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

Demonstrate that fine-tuning improved behavior compared to the base model.

### Required Evaluation

1. Create **10–15 custom test prompts**
   - Must not be copied from the dataset
   - Should reflect real customer support scenarios

2. Compare:
   - Base model responses
   - Fine-tuned model responses

3. Save a **side-by-side comparison** showing:
   - Prompt
   - Base output
   - Fine-tuned output
   - Short commentary on differences

Focus on:

- Helpfulness
- Tone and professionalism
- Specificity (less generic responses)

## Step 5: Demo (Optional but Encouraged)

Create a simple demo using **Gradio** that:

- Loads the base model
- Applies your fine-tuned adapter
- Accepts user input
- Displays the model’s response

This demonstrates real-world usability.
