import faiss
import os
import numpy as np
import streamlit as st
import math
from torchvision.io import read_image
import torch

def build_faiss(progress_bar, database_path, model, preprocess, d, device):
    nb = len(os.listdir(database_path))
    xb = []
    index = faiss.IndexFlatL2(d)
    model.eval().to(device)
    img_names = []
    
    for i, img_name in enumerate(os.listdir(database_path)):
        progress_bar.progress(math.ceil(i * 100 / nb), "Building database indexing")
        img_names.append(img_name)
        img_path = os.path.join(database_path, img_name)
        img = read_image(img_path)
        with torch.no_grad():
            try:
                batch = preprocess(img).unsqueeze(0).to(device)
                embedding = model(batch)['output'].squeeze(0).detach().cpu().numpy()
                xb.append(embedding)
            except Exception as error:
                # st.write(error)
                continue
    
    xb = np.array(xb, dtype=np.float32)
    index.add(xb)
    
    return index, img_names

def checkDatabasePath(path):
    if len(os.listdir(path)) == 0:
        st.sidebar.warning("Folder contain no file! Please choose again")
        st.stop()
    img_names = [x for x in os.listdir(path) if x.split(".")[-1] in ['jpg', 'jpeg', 'png']]
    if len(img_names) == 0:
        st.sidebar.warning("Folder contain no image! Please choose again")
        st.stop()
    return int(len(img_names))