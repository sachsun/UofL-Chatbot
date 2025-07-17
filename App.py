import json
import threading
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from openai import OpenAI, OpenAIError
import logging
import sys
import os

# —————— CONFIG ——————
API_KEY          = ""
MODEL            = "gpt-3.5-turbo"
EMBED_MODEL      = "text-embedding-ada-002"
FAQ_FILE         = "uofl_faq_100.json"
LOGO_FILE        = "cardinal_logo.png"
EMBED_CACHE_FILE = "faq_embeddings.npy"

# —————— LOGGING SETUP ——————
logging.basicConfig(
    filename="cardinal_guide.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

client = OpenAI(api_key=API_KEY)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# —————— LOAD FAQ DATA ——————
with open(resource_path(FAQ_FILE), "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# —————— EMBEDDING HELPERS ——————
def embed_text(text: str) -> list[float]:
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=[text]
        )
        return resp.data[0].embedding
    except OpenAIError:
        logging.error("Embedding API error for text: %s", text, exc_info=True)
        raise

def load_or_compute_embeddings():
    if os.path.exists(EMBED_CACHE_FILE):
        try:
            return np.load(EMBED_CACHE_FILE, allow_pickle=True).tolist()
        except Exception:
            logging.warning("Failed to load embeddings cache; recomputing", exc_info=True)

    embeddings = []
    for item in faq_data:
        try:
            embeddings.append(embed_text(item["question"]))
        except Exception:
            logging.error("Failed to embed FAQ question: %s", item["question"], exc_info=True)
            embeddings.append([0.0] * 1536)  # fallback zero-vector
    np.save(EMBED_CACHE_FILE, embeddings, allow_pickle=True)
    return embeddings

faq_embeddings = load_or_compute_embeddings()

def top_k_faqs(query: str, k: int = 3) -> list[dict]:
    q_emb = embed_text(query)
    sims = [
        np.dot(q_emb, f_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(f_emb))
        for f_emb in faq_embeddings
    ]
    best_idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [faq_data[i] for i in best_idxs]

# —————— RAG + STRICT ON‑TOPIC CHAT ——————
def get_response(prompt: str) -> str:
    try:
        relevant = top_k_faqs(prompt, k=3)
        context = "\n\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in relevant)

        system_msg = (
    "You are **Cardinal Guide**, a caring assistant *only* for University of Louisville students.\n"
    "Answer **only** UofL questions — like services, buildings, registration, dining, parking, clubs, or campus support.\n"
    "If someone asks something not about UofL, respond casually like: 'Oops, I can’t help with that — but I can totally answer anything about UofL!'\n\n"
    "Context:\n" + context
)
    
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt}
            ],
            timeout=15
        )
        return resp.choices[0].message.content.strip()

    except OpenAIError:
        logging.error("OpenAI API error in get_response", exc_info=True)
        return "Sorry, I’m having a bit of trouble right now—please try again in a minute."
    except Exception:
        logging.exception("Unexpected error in get_response")
        return "Something went wrong; please try again later."

# —————— TKINTER UI WITH LOGO AVATAR ——————
class CardinalGuideApp:
    def __init__(self, root):
        root.title("Cardinal Guide — UofL Assistant")
        root.geometry("800x600")

        logo_img = Image.open(resource_path(LOGO_FILE)).resize((24, 24), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_img)
        root.iconphoto(False, self.logo_photo)

        self.chat_window = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state='disabled', width=80, height=25, bg="#FFFFFF"
        )
        self.chat_window.pack(padx=10, pady=10)

        self.entry = tk.Entry(root, width=80)
        self.entry.pack(padx=10, pady=(0, 10))
        self.entry.bind("<Return>", self.on_send)

        self.send_btn = tk.Button(root, text="Send", command=self.on_send)
        self.send_btn.pack()

        self._insert_message(
            "Cardinal Guide",
            "Hey there! I’m Cardinal Guide — your go-to helper for anything UofL. What can I help you with today?"
        )

    def _insert_message(self, sender: str, text: str):
        self.chat_window.config(state='normal')
        if sender == "Cardinal Guide":
            self.chat_window.image_create(tk.END, image=self.logo_photo)
            self.chat_window.insert(tk.END, " ")
        self.chat_window.insert(tk.END, f"{sender}: {text}\n\n")
        self.chat_window.config(state='disabled')
        self.chat_window.yview(tk.END)

    def on_send(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, tk.END)
        self._insert_message("You", user_text)
        threading.Thread(target=self._get_and_insert, args=(user_text,), daemon=True).start()

    def _get_and_insert(self, prompt: str):
        reply = get_response(prompt)
        self._insert_message("Cardinal Guide", reply)

if __name__ == "__main__":
    root = tk.Tk()
    app = CardinalGuideApp(root)
    root.mainloop()
