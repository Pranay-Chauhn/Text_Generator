import os
from transformers import GPT2Tokenizer
from datasets import Dataset

# Constants
CORPUS_PATH = "app/preprocessing/cleaned_corpus.txt"
SEQ_LEN = 1024
MODEL_NAME = "gpt2"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load corpus
with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize entire text (no truncation)
print("🔄 Tokenizing text...")
tokens = tokenizer(text, return_tensors='pt', truncation=False)["input_ids"][0]
print(f"✅ Total tokens: {len(tokens)}")

# Chunk into sequences (convert to list of lists of ints)
print("🔄 Splitting into chunks...")
chunks = [tokens[i:i+SEQ_LEN].tolist() for i in range(0, len(tokens)-SEQ_LEN, SEQ_LEN)]
dataset_dict = {"input_ids": chunks}

# Create Dataset
hf_dataset = Dataset.from_dict(dataset_dict)

# Save to disk
output_dir = "app/model/tokenized_data"
os.makedirs(output_dir, exist_ok=True)
hf_dataset.save_to_disk(output_dir)

print(f"✅ Tokenized dataset saved at {output_dir}")
