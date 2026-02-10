from datasets import load_dataset
import pandas as pd

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

print(ds["train"][5])
df = pd.DataFrame(ds["train"])

print(f"Total examples: {len(df)}")
print("Columns:", df.columns)
print("Sample row:")
print(df.iloc[0])

