from typing import Any
import requests
from bs4 import BeautifulSoup
import re
import json, numpy as np, faiss
import ollama
import os
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

if os.path.exists("zoning_r6.txt"):
    text = open("zoning_r6.txt", encoding="utf-8").read()
else:
    r = requests.get("https://zr.planning.nyc.gov/article-i/chapter-1")
    r.raise_for_status()  # surface HTTP errors early

    soup = BeautifulSoup(r.text, "html.parser")
    element = soup.select_one("#block-neoclassic-content")

    if element is None:
        raise RuntimeError("Selector matched nothing — the page structure changed.")
    else:
        text = element.get_text(separator="\n", strip=True)
        with open("zoning_r6.txt", "w", encoding="utf-8") as f:
            f.write(text)

def find_source_url(line, i, num):
    """Return (url, index) for the source link near section start i, else (None, None)."""
    for j in range(i + 1, min(i + 9, len(line))):
        if line[j].startswith("http") and line[j].endswith("/" + num):
            return line[j], j
    return None, None


def is_real_section_start(line, i):
    num = line[i].strip()

    # TODO 1: must look like a section number, e.g. "11-121"
    if not re.match(r"^\d+-\d+$", num):
        return False

    # TODO 2: a real header is followed by a URL ending in "/<num>"
    url, _ = find_source_url(line, i, num)
    if url is not None:
        return True

    # TODO 3: no matching URL nearby -> not a real section start
    return False


lines = [ln.strip() for ln in text.split("\n")]

# indices where a real section begins
starts = [i for i in range(len(lines)) if is_real_section_start(lines, i)]

sections = []
for k, i in enumerate(starts):
    num = lines[i]
    title = lines[i + 1] if i + 1 < len(lines) else ""
    url, url_idx = find_source_url(lines, i, num)

    # body runs from the line after the URL to the start of the next section
    body_start = (url_idx + 1) if url_idx is not None else (i + 2)
    body_end = starts[k + 1] if k + 1 < len(starts) else len(lines)
    body = "\n".join(lines[body_start:body_end]).strip()

    sections.append({
        "number": num,
        "title": title,
        "url": url,
        "body": body,
    })

sections = [s for s in sections if s["body"]]

def make_chunk(s, text):
    return {
        "number": s["number"],
        "title": s["title"],
        "url": s["url"],
        "text": text,
    }

CHUNKS_FILE = "zoning_chunks.json"

if os.path.exists(CHUNKS_FILE):
    chunks = json.load(open(CHUNKS_FILE, encoding="utf-8"))
else:
    MAX = 3000
    OVERLAP = 1
    chunks = []
    for s in sections:
        body = s["body"]

        if len(body) <= MAX:
            chunks.append(make_chunk(s, body))
            continue

        buffer = []
        size = 0
        for line in body.split("\n"):
            # +1 accounts for the "\n" that will rejoin the lines
            if buffer and size + len(line) + 1 > MAX:
                chunks.append(make_chunk(s, "\n".join(buffer)))
                buffer = buffer[-OVERLAP:]  # seed next buffer for overlap
                size = sum(len(l) + 1 for l in buffer)
            buffer.append(line)
            size += len(line) + 1

        if buffer:
            chunks.append(make_chunk(s, "\n".join(buffer)))
    json.dump(chunks, open(CHUNKS_FILE, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"Built and saved {len(chunks)} chunks")

# TODO 1: load the model
model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = json.load(open("zoning_chunks.json", encoding="utf-8"))

# TODO 1: build `texts` = a list of (title + "\n" + body) for each chunk
texts = [c["title"] + "\n" + c["text"] for c in chunks]

# TODO 2: encode `texts` (reusing the model loaded above), normalized for cosine/IP
embeddings = model.encode(texts, normalize_embeddings=True)

# TODO 3: convert to a float32 numpy array (faiss requires float32)
embeddings = np.asarray(embeddings, dtype="float32")

# TODO 4: build an inner-product index (== cosine sim on normalized vectors)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)
def generate(system_prompt, user_content):
    resp = client.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    return resp["message"]["content"]

def ask(query: object, k: object = 3) -> Any:
    # TODO 1: embed the query the same way as the index (normalized, float32)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    # TODO 2: search the index for the k nearest chunks
    scores, idxs = index.search(q, k)

    # TODO 3: return each hit's chunk together with its similarity score
    hits = []
    for pos, score in zip(idxs[0], scores[0]):
        hits.append((chunks[pos], float(score)))

    context = "\n\n".join(
        f"[{c['number']} {c['title']}] ({c['url']})\n{c['text']}" for c, _ in hits
    )
    system_prompt = ("You answer questions about the NYC Zoning Resolution using ONLY the provided excerpts. "
                     "Cite the section number for every claim. If the excerpts don't contain the answer, say so.")
    user_content = f"Excerpts:\n\n{context}\n\nQuestion: {query}"
    answer = generate(system_prompt, user_content)
    return {"answer": answer, "sources": [c["url"] for c, _ in hits]}

if __name__ == "__main__":
    ask("How are zoning districts named?")   # manual testing only

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask_endpoint(req: Query):
    return ask(req.query)  