import requests
import os

# Create the books directory inside app/
base_dir = "app"
books_dir = os.path.join(base_dir,"books")
os.makedirs(books_dir, exist_ok=True)


def download_books(book_url, save_path, start_marker = '*** START', end_marker = '***END') :
    try :
        response = requests.get(book_url)
        response.raise_for_status()
    except Exception as e :
         print(f"Failed to download {book_url} : {e}")
         return

    text = response.text
    start = text.find(start_marker)
    end = text.find(end_marker)
    cleaned_text = text[start:end].strip()


    with open(save_path, "w", encoding='utf-8') as f :
        f.write(cleaned_text)
    

    print(f"Saved : {save_path}")
    return

# Dictionary of books and their URLs
books = {
    "Pride_and_Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "Frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "Alice_in_Wonderland": "https://www.gutenberg.org/files/11/11-0.txt"
}

# Download each books
for title, url in books.items() :
    filename = os.path.join(books_dir, f"{title}.txt")
    download_books(url,filename)
