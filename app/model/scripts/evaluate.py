from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math

# Load fine-tuned model
model_path = "app/model/gpt2_finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model.eval()

# Load cleaned corpus for evaluation
with open("app/preprocessing/cleaned_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()[:1000]  # use first 1000 chars to compute perplexity quickly

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss).item()

print(f"🧠 Perplexity: {perplexity:.2f}")
