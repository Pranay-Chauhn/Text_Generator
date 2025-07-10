#  Fine-tuned GPT-2 Text Generator (with Streamlit UI)

This project fine-tunes OpenAI’s GPT-2 on classical literature using Hugging Face Transformers, and provides an interactive text generation interface via Streamlit.

---

## ✨ Project Goal

To build a generative model that creates grammatically correct, creative text based on input prompts.

---

## 🗂️ Dataset

- 📖 Source: [Project Gutenberg](https://www.gutenberg.org/)
- Books Used:
  - *Pride and Prejudice* by Jane Austen
  - *Frankenstein* by Mary Shelley
  - *Alice in Wonderland* by Lewis Carroll

---

## 🔧 Pipeline

| Step | Description |
|------|-------------|
| 1. Data Collection | Downloads `.txt` files using `requests` |
| 2. Preprocessing | Cleans and sentence-tokenizes using `re` + `nltk` |
| 3. Tokenization | Converts cleaned text into GPT-2 token sequences |
| 4. Model Training | Fine-tunes GPT-2 using HuggingFace `Trainer` API |
| 5. Evaluation | Reports perplexity and human-reviewed samples |
| 6. Text Generation | Generates creative output from prompts |
| 7. UI | Interactive Streamlit app to test generation |

---

## 🧪 Training Details

- Model: `gpt2` (117M parameters)
- Epochs: 3
- Token Length: 1024
- Batch Size: 2
- Optimizer: AdamW
- Frameworks: PyTorch, Hugging Face Transformers, Datasets

---

## 📊 Evaluation

- **Perplexity**: ~`XX.XX` *(add your score here)*
- **Sample Output**:

## Tech-Stack :
- **Transformers**
- **Datasets**
- **NLTK**
- **Streamlit**
- **PyTorch**

## Author
- **Name:** Pranay Chauhan
- **Focus:** NLP, GenAI, Data Science

## 🚀 Streamlit Interface

### 🔧 How to Run:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run main.py


