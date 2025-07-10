# main.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# App title
st.set_page_config(page_title=" GPT-2 Text Generator")
st.title(" Fine-tuned GPT-2 Text Generator")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "D:\\Projects\\Text_Gen\\app\\model\\gpt2_finetuned"
    if not os.path.exists(model_path):
        st.error("Model not found! Please train the model first.")
        return None, None
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Prompt input Enter a prompt to start generating text:", "Once upon a time")

if st.button(" Generate Text") and model:
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with st.spinner("Generating..."):
        output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.90,
        do_sample=True,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        repetition_penalty=1.2)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        # Save output
        os.makedirs("app/output", exist_ok=True)
        with open("app/output/generated_sample.txt", "w", encoding="utf-8") as f:
            f.write(generated)

    st.subheader(" Generated Text")
    st.text_area(label="", value=generated, height=300)

    st.success(" Text generation complete!")
