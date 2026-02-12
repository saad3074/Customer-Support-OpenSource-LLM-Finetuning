pip install -r requirements.txt

# Login to Hugging Face (for Llama model access)
huggingface-cli login

# Quick test on 500 examples
python main.py train --subset 500 --output_dir output/customer-support-llm

# Full training
python main.py train --output_dir output/customer-support-llm

# Run evaluation
python main.py evaluate --adapter_path output/customer-support-llm

# Start Gradio demo
python main.py demo --adapter_path output/customer-support-llm




# Since you already have gemma3:12b pulled in Ollama on localhost,
# Demo (chat UI):
python main.py demo --ollama



# Uses gemma3:12b by default. To be explicit:
python main.py demo --ollama --ollama-model gemma3:12b


# Evaluate (run test prompts and save to JSON):
python main.py evaluate --ollama --ollama-model gemma3:12b --output ollama_gemma_eval.json

# Or with Hugging Face adapter (writes evaluation_results.json by default):
# python main.py evaluate --adapter_path output/customer-support-llm

# Generate evaluation report — run one of the evaluate commands above first, then:
python main.py report --evaluation evaluation_results.json --output evaluation_report.html

# With training stats (LoRA/QLoRA + loss curve):
python main.py report --evaluation evaluation_results.json --training_stats output/customer-support-llm/training_stats.json --output evaluation_report.html

# If you used a different eval output file, pass it: --evaluation ollama_gemma_eval.json

# Check evaluation: open evaluation_results.json (raw) or evaluation_report.html in a browser



