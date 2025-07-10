import os
import re
import nltk 
nltk.download('punkt')  # for sentence tokenizer
from nltk.tokenize import sent_tokenize

#Paths
books_dir  = "app/books"
output_path = "app/preprocessing/cleaned_corpus.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def clean_text(text):
    # Cleaning raw text
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"()\s]','',text)
    return text.strip()


def preprocess_books(book_folder) :
    """ loading all books, 
        cleaning text, 
        tokenize into sentence, 
        and joining into a large corpus
    """
    corpus = []

    for filename in os.listdir(book_folder) :
        if filename.endswith(".txt") :
            with open(os.path.join(book_folder,filename), 'r', encoding='utf-8') as f :
                raw = f.read()
                cleaned = clean_text(raw)
                sentence = sent_tokenize(cleaned)
                corpus.extend(sentence)
    # join into long string, one sentence per line
    return "\n".join(corpus)

# Run Preprocessing
print("Processing books...")
final_text = preprocess_books(books_dir)

# Save cleaned corpus
with open(output_path,"w", encoding='utf-8') as f :
    f.write(final_text)


print(f"Preprocessing complete! Saved to : {output_path}")

