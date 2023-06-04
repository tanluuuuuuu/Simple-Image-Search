import clip
import torch
from PIL import Image
import os
import faiss
import math
import numpy as np
import streamlit as st

def init_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def build_faiss_clip(progress_bar, database_path, model, preprocess, device):
    d = 512
    nb = len(os.listdir(database_path))
    xb = []
    index = faiss.IndexFlatL2(d)
    model.eval().to(device)
    img_names = []
    for i, img_name in enumerate(os.listdir(database_path)):
        progress_bar.progress(math.ceil(i * 100 / nb), "Building clip indexing")
        img_names.append(img_name)
        img_path = os.path.join(database_path, img_name)
        img = Image.open(img_path)
        with torch.no_grad():
            try:
                batch = preprocess(img).unsqueeze(0).to(device)
                embedding = model.encode_image(batch).squeeze().detach().cpu().numpy()
                xb.append(embedding)    
            except Exception as error:
                # st.write(error)
                continue
    xb = np.array(xb)
    index.add(xb)
    
    return index, img_names

def return_text_embedding(search_key, model, device):
    text = clip.tokenize([search_key]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()
    return text_features