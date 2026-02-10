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