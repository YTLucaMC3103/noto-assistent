import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os

# Lade dein Embedding-Modell (klein & schnell)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lade OpenAI API-Key aus Umgebungsvariable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialisiere lokale Chroma-DB
client = chromadb.Client()
collection = client.get_or_create_collection("wissen")

# Fülle Vektor-DB beim Start (nur beim ersten Mal nötig)
def lade_daten():
    try:
        with open("faq.txt", "r") as f:
            zeilen = f.readlines()

        for i, zeile in enumerate(zeilen):
            collection.add(
                documents=[zeile],
                embeddings=[model.encode(zeile).tolist()],
                ids=[f"id_{i}"]
            )
    except:
        pass  # Fehler ignorieren, falls schon geladen

lade_daten()

# 🔍 Relevante Inhalte aus Vektordatenbank holen
def suche_relevante_passagen(user_input):
    user_vec = model.encode(user_input).tolist()
    results = collection.query(query_embeddings=[user_vec], n_results=3)
    return results["documents"][0]  # Liste mit Strings

# 💬 GPT-Antwort generieren
def gpt_antwort(user_input, kontext_passagen):
    prompt = f"""Du bist Beet, ein freundlicher, direkter Assistent für noto.
Nutze dieses Wissen, um die Frage zu beantworten:
{kontext_passagen}

Frage: {user_input}
Antwort:"""

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    return res.choices[0].message.content
