# Customer Support LLM Fine-Tuning: Complete Plan & Data Understanding

## Executive Summary

This document provides a **deep understanding** of the Bitext Customer Support dataset, a **mapping strategy** from raw data to the fine-tuning task, a **data preparation pipeline** for QLoRA training, and a **generalization strategy** to ensure the model works on any test data—including queries not seen during training.

---

## Part 1: Deep Understanding of the Dataset

### 1.1 Dataset Source & Metadata

| Attribute | Value |
|-----------|-------|
| **Source** | [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| **Provider** | Bitext (Hugging Face) |
| **License** | CDLA-Sharing 1.0 |
| **Language** | English |
| **Size** | ~26,900 rows |
| **Split** | Single `train` split only |
| **Format** | CSV (auto-converted to Parquet on HF) |
| **Tasks** | Question Answering, Conversational AI |

### 1.2 Schema & Column Structure

The dataset has **5 columns**:

| Column | Type | Description | Key Characteristics |
|--------|------|-------------|---------------------|
| **instruction** | string | Customer query / user utterance | 6–92 chars; natural language variations |
| **response** | Agent reply / target output | 57–2,470 chars; structured, professional |
| **category** | string | High-level domain (11 values) | e.g., ORDER, SHIPPING, PAYMENT |
| **intent** | string | Specific intent (27 values) | e.g., cancel_order, track_order, refund |
| **flags** | string | Unique identifier (394 values) | Likely example/variant IDs |

### 1.3 Semantic Structure of the Data

#### Instructions (Customer Queries)

- **Paraphrase-rich**: Same intent expressed in many ways (e.g., “I need to cancel order X”, “help me canceling purchase X”, “can you help me cancel order X?”).
- **Placeholder tokens**: Use `{{Order Number}}`, `{{Online Company Portal Info}}`, `{{Customer Support Hours}}`, etc. These are **generic placeholders** for company-specific values.
- **Typos & informal language**: Some examples include “oorder”, “pucrhase”, “canceling” vs “cancelling”.
- **Emotional variation**: From polite (“Could you help me…”) to frustrated (“help me to cancel my last goddamn purchase”).

#### Responses (Agent Replies)

- **Structured format**: Numbered steps, clear sections, consistent closing.
- **Professional tone**: Empathetic, solution-oriented, apologetic when appropriate.
- **Placeholder-heavy**: Same `{{...}}` tokens for company-specific info (portal URL, phone, hours, etc.).
- **Length variation**: Short acknowledgments (57 chars) to detailed multi-step guides (2,470 chars).

#### Category & Intent

- **category**: Broad domain (ORDER, etc.).
- **intent**: Fine-grained action (cancel_order, track_order, refund, etc.).
- **Use**: Useful for analysis, filtering, and stratified splits—not required for chat formatting.

### 1.4 Data Distribution Insights

From the Hugging Face Dataset Viewer:

- **Intent coverage**: 27 intents across 11 categories.
- **Paraphrase density**: Many instructions per intent (e.g., 50+ variants for `cancel_order`).
- **Response templates**: Multiple response templates per intent with different phrasings but same structure.
- **Imbalance**: Some intents likely have more examples than others.

### 1.5 Placeholder Tokens (Critical for Generalization)

The dataset uses **abstract placeholders** instead of real company data:

| Placeholder | Purpose |
|-------------|---------|
| `{{Order Number}}` | Order/purchase identifier |
| `{{Online Company Portal Info}}` | Login URL / account portal |
| `{{Online Order Interaction}}` | UI labels (e.g., “Order History”, “Cancel Order”) |
| `{{Customer Support Hours}}` | Support availability |
| `{{Customer Support Phone Number}}` | Support phone |
| `{{Website URL}}` | Company website |
| `{{Company}}` | Company name |
| `{{Cancel Purchase}}` | Cancel button/link label |
| `{{Cancellation Policy}}` | Policy reference |

**Why this matters**: The model learns **patterns and structure**, not fixed strings. At inference, you can either:
- Replace placeholders with real values via post-processing, or
- Let the model output placeholders and replace them in your application layer.

---

## Part 2: Mapping Data to the Fine-Tuning Task

### 2.1 Task Definition

**Goal**: Fine-tune Llama 3.2 3B with QLoRA so it acts as a **customer support assistant** that:
- Responds clearly, professionally, and accurately to customer queries.
- Handles order management, shipping, payments, returns, etc.
- Generalizes to new phrasings and intents similar to those in the training set.

### 2.2 Mapping Strategy

| Raw Field | Role in Training | Notes |
|-----------|------------------|-------|
| `instruction` | **User message** | Direct use as customer query |
| `response` | **Assistant message** | Target output for the model |
| `category` | Optional metadata | For filtering, stratification, or analysis |
| `intent` | Optional metadata | For evaluation, stratified splits |
| `flags` | Optional | For deduplication or tracking |

### 2.3 Chat Format Mapping

Each example becomes a **single-turn conversation**:

```
[System] You are a helpful customer support assistant.
[User]   <instruction>
[Assistant] <response>
```

This matches the instruction-following / chat format expected by Llama 3.2 and SFTTrainer.

### 2.4 What the Model Learns

1. **Response structure**: Step-by-step instructions, empathetic openings, contact info sections.
2. **Tone**: Professional, helpful, apologetic when appropriate.
3. **Intent handling**: How to respond to cancel, track, refund, etc.
4. **Placeholder usage**: When and where to use `{{...}}` tokens (or you can strip them; see Section 4).

---

## Part 3: Creating Data for Fine-Tuning

### 3.1 Data Preparation Pipeline

```
Raw Dataset (HF) → Load → Format → (Optional) Split → Train
```

#### Step 1: Load

```python
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
```

#### Step 2: Format to Chat

```python
def format_as_chat(example):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
    }

dataset = dataset.map(format_as_chat, remove_columns=dataset["train"].column_names)
```

#### Step 3: Train/Validation Split

- Dataset has **no validation split**.
- **Action**: Create one (e.g., 95% train / 5% validation) via `dataset.train_test_split(test_size=0.05, seed=42)`.
- **Recommendation**: Use **stratified split** by `intent` to ensure all intents appear in both splits.

#### Step 4: Optional Filtering

- Remove very short or very long examples if they cause training issues.
- Filter by `category` or `intent` if you want to focus on a subset.

### 3.2 Handling Placeholders

**Option A – Keep placeholders (recommended)**  
- Model learns to output `{{Order Number}}`, `{{Website URL}}`, etc.
- At inference: post-process to replace with real values from your system.

**Option B – Replace with generic text**  
- Replace `{{Order Number}}` with “your order number” before training.
- Simpler outputs, but less flexible for dynamic content.

**Option C – Remove placeholders**  
- Strip `{{...}}` and leave blanks or generic text.
- Use only if you plan to inject values entirely at inference.

### 3.3 Sequence Length Considerations

- **Instruction**: typically 6–92 chars.
- **Response**: 57–2,470 chars.
- **Recommendation**: `max_seq_length` 512–1024 (per instructions). Some long responses may be truncated; monitor loss on long examples.

### 3.4 Training Configuration Summary

| Parameter | Suggested Value |
|-----------|-----------------|
| `max_seq_length` | 512–1024 |
| `batch_size` | 1–4 |
| `gradient_accumulation_steps` | 4–8 |
| `epochs` | 1 |
| `seed` | 42 |
| Test subset first | 100–500 examples |

---

## Part 4: Ensuring It Works on Any Test Data

### 4.1 The Generalization Challenge

Training data covers:
- 27 intents
- Many paraphrases per intent
- Placeholder-based, domain-agnostic responses

Test data may include:
- **In-domain, unseen phrasings**: Same intent, new wording.
- **Out-of-domain intents**: Completely new request types.
- **Edge cases**: Typos, mixed languages, very long queries.

### 4.2 Strategies for Generalization

#### 4.2.1 Paraphrase Coverage (Already in Data)

- Dataset has many paraphrases per intent.
- Model should generalize to similar phrasings.
- **Action**: Include diverse paraphrases in training; avoid over-sampling a single template.

#### 4.2.2 Intent Coverage

- 27 intents cover common support scenarios.
- **Action**: Ensure validation set has examples from all intents.
- **Limitation**: Truly new intents (e.g., “I want to speak to a human”) may need fallback behavior.

#### 4.2.3 Fallback / Out-of-Scope Handling

- Add a small set of **out-of-scope** examples:
  - User: “Tell me a joke” → Assistant: “I’m here to help with orders, shipping, and account questions. How can I assist you?”
- This teaches the model to redirect instead of hallucinating.

#### 4.2.4 Placeholder Handling at Inference

- **Pipeline**: User query → Model → Raw output → Replace `{{X}}` with real values from DB/config.
- Ensures the same model works across different companies by swapping placeholders.

#### 4.2.5 Evaluation on Held-Out Prompts

- Create **10–15 custom test prompts** (per instructions) that are **not** in the dataset.
- Cover: order cancel, track, refund, account, shipping, edge cases.
- Compare base vs fine-tuned model on:
  - Helpfulness
  - Tone and professionalism
  - Specificity (less generic)

### 4.3 Test Data Design

| Test Type | Example | Purpose |
|-----------|---------|---------|
| Paraphrase | “I wanna cancel #12345” | Same intent, new wording |
| Typo | “I need to cancle my order” | Robustness |
| New structure | “Order 12345 – please cancel it” | Structural variation |
| Edge case | “Cancel everything” | Ambiguity handling |
| Out-of-scope | “What’s the weather?” | Fallback behavior |

### 4.4 Monitoring & Iteration

1. **Qualitative review**: Manually inspect 20–30 model outputs.
2. **Quantitative**: Track loss on validation set.
3. **A/B comparison**: Base vs fine-tuned on custom prompts.
4. **Iterate**: Add more training examples for weak intents or add fallback examples.

---

## Part 5: End-to-End Plan Summary

| Phase | Action |
|-------|--------|
| **1. Data Understanding** | Inspect schema, distribution, placeholders, intent coverage |
| **2. Data Preparation** | Load → format as chat → stratified train/val split |
| **3. Training** | QLoRA on Llama 3.2 3B, 1 epoch, small batch, test on subset first |
| **4. Evaluation** | 10–15 custom prompts, side-by-side base vs fine-tuned |
| **5. Generalization** | Fallback examples, placeholder post-processing, diverse test set |
| **6. Demo** | Gradio app loading base + adapter, user input → model response |

---

## Appendix: Quick Reference

### Dataset Load & Format (Minimal)

```python
from datasets import load_dataset

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
ds = ds["train"].train_test_split(test_size=0.05, seed=42)

def format_example(ex):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]}
        ]
    }

ds = ds.map(format_example, remove_columns=ds["train"].column_names)
```

### Key Placeholders to Handle at Inference

- `{{Order Number}}` → actual order ID
- `{{Online Company Portal Info}}` → login URL
- `{{Customer Support Hours}}` → support hours
- `{{Customer Support Phone Number}}` → support phone
- `{{Website URL}}` → company website

---

*This plan is based on the Bitext Customer Support dataset structure as of the latest Hugging Face metadata and dataset viewer.*
