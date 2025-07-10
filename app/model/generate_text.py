from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model & tokenizer
model_path = "app/model/gpt2_finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model.eval()

# User prompt
prompt = input("📝 Enter a prompt: ")

# Encode prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    early_stopping=True
)

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n📜 Generated Text:\n")
print(generated_text)

# Save to file
with open("app/output/generated_sample.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)
