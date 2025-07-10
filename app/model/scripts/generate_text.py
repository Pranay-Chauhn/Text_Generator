from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model & tokenizer
model_path = "app/model/gpt2_finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model.eval()

# User prompt
prompt = input("📝 Enter a prompt: ")

def gpt_response(prompt,max_length=150,temperature=0.7) :
    
    #Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=temperature,
        top_k=50,
        top_p=0.90,
        do_sample=True,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        repetition_penalty=1.2
    )

    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n📜 Generated Text:\n")
    print(generated_text)

    # Save to file
    with open("app/output/generated_sample.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
    return generated_text

gpt_response(prompt)