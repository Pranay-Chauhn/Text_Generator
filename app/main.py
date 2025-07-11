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
    try :
        model_path = "Pranay-Chauhn/gpt2-finetuned"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.eval()
        return tokenizer, model
    except Exception as e :
        st.error(f"failed to load the model: {e} ")
        return None, None
tokenizer, model = load_model()

# Prompt input
prompt = st.text_input(" Enter a prompt to start generating text:", "Once upon a time")

if st.button(" Generate Text") and model:
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with st.spinner("Generating..."):
        output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        temperature=0.6,
        top_k=40,
        top_p=0.85,
        do_sample=True,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        repetition_penalty=1.3
        )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)

    st.subheader(" Generated Text")
    st.text_area(label="", value=generated, height=300)

    st.success(" Text generation complete!")
