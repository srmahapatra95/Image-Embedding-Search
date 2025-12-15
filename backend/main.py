import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File
import open_clip
import torch 
from PIL import Image
import numpy as np
import faiss

app = FastAPI()

# Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Load the clip model
model, preprocess = open_clip.create_model_from_pretrained("ViT-B-32", pretrained="openai")
model.to(DEVICE)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Vector index
embedding_dimension = 512
index = faiss.IndexFlatL2(embedding_dimension)
image_store = []

def get_embedding(image: Image.Image):
    image = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

@app.get("/")
def home():
    return {"message": "FastAPI with reload is working!"}

@app.post("/add-image/")
async def add_image(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    emb = get_embedding(img)
    index.add(emb)

    image_store.append({"filename": file.filename})
    return {"status": "stored"}

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    if index.ntotal == 0:
        return {"results": [], "message": "No images in index"}

    img = Image.open(file.file).convert("RGB")
    query_emb = get_embedding(img)

    k = min(5, index.ntotal)
    D, I = index.search(query_emb, k=k)
    results = [image_store[i] for i in I[0] if i >= 0]
    return {"results": results}

