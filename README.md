# Fine-Tune an Open-Source Customer Support LLM (QLoRA)

## Project Overview

In this project, you will fine-tune an **open-source LLM (Llama 3.2 3B)** using **QLoRA** to build a customer support chatbot. You will work with a real-world dataset, train and evaluate your model.

This assignment is designed to mirror a real industry fine-tuning workflow using modern open-source tooling.

## Learning Objectives

By completing this project, you will:

- Set up an environment for fine-tuning open-source LLMs
- Prepare and format real-world instruction-style datasets
- Fine-tune a model using **QLoRA (4-bit + LoRA adapters)**
- Evaluate a fine-tuned model against a base model
- Run inference and demonstrate results with a small demo

## Project Scenario

You are tasked with building a **customer support chatbot** trained on historical support conversations.

**Model**: Llama 3.2 3B  
**Technique**: QLoRA  
**Dataset**: Bitext Customer Support Dataset (Hugging Face)

The goal is to improve the model’s ability to respond clearly, professionally, and accurately to customer queries compared to the base model.

## What You Need to Do

You will:

- Prepare the dataset for instruction/chat-style training
- Fine-tune the model using QLoRA
- Evaluate performance (qualitative + basic quantitative)
- Run inference on custom prompts
- Add a simple demo (e.g., Gradio)

All **step-by-step instructions**, including environment setup and code guidance, are provided in **`instructions.md`**.
