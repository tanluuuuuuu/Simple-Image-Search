import streamlit as st
from PIL import Image
from io import BytesIO
import os
import torch
import tkinter as tk
from tkinter import filedialog

from utils import *
from model_embedding import *
from clip_embedding import *

# Initialize session states
def init_states():
    if 'fs_idx' not in st.session_state:
        st.session_state['fs_idx'] = None
        st.session_state['img_idx'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
        st.session_state['preprocess'] = None
        st.session_state['dimension'] = None
    if 'db_path' not in st.session_state:
        st.session_state['db_path'] = None
    if 'clip_model' not in st.session_state:
        st.session_state['clip_model'] = None    
        st.session_state['clip_preprocess'] = None 
    if 'fs_idx_clip' not in st.session_state:     
        st.session_state['fs_idx_clip'] = None
        st.session_state['img_idx_clip'] = None
    
# Reset model-related session states
def onChangeModel():
    st.session_state['model'] = None
    st.session_state['preprocess'] = None
    st.session_state['dimension'] = None
    st.session_state['fs_idx'] = None  
    st.session_state['img_idx'] = None
    
def onChangeDevice():
    st.empty()
    pass

if __name__ == '__main__':
    # Initialize session states 
    init_states()
    
    # device (CPU or CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)
    
    # Set the path to the image database
    st.sidebar.title("Image Search")
    st.sidebar.write("Device: ", device)
    st.sidebar.write('Please select a folder:')
    col_folder = st.sidebar.columns(2)
    clicked_choose_folder = col_folder[0].button('Folder Picker')
    clicked_sample_path = col_folder[1].button('Use sample database')
    db_path = st.session_state['db_path']
    if clicked_choose_folder:
        db_path = st.sidebar.text_input('Selected folder database:', filedialog.askdirectory(master=root))
    elif clicked_sample_path:
        current_directory = os.path.abspath(os.getcwd())
        db_path = os.path.join(current_directory, 'sample_data')
        db_path = st.sidebar.text_input('Selected folder database:', value=db_path)
    num_database_images = checkDatabasePath(db_path)
    if st.session_state['db_path'] != db_path:
        st.session_state['db_path'] = db_path
        st.session_state['clip_model'] = None 
        st.session_state['model'] = None
        st.session_state['fs_idx'] = None
    if st.session_state['db_path'] == None:
        st.write("Select database path to continue")
        st.stop()
    else:
        st.sidebar.success(f"Found {num_database_images} images in database")

    # Build clip model
    if st.session_state['clip_model'] == None:
        progress_bar = st.progress(0, text="Download weight and building a text model")
        st.session_state['clip_model'], st.session_state['clip_preprocess'] = init_clip_model(device) 
        st.session_state['fs_idx_clip'], st.session_state['img_idx_clip'] = build_faiss_clip(progress_bar, 
                                                                                             st.session_state['db_path'],
                                                                                             st.session_state['clip_model'],
                                                                                             st.session_state['clip_preprocess'],device)
        progress_bar.empty()
    
    # Choose the model and number of output images
    model_name = st.sidebar.selectbox('Choose model', ('ViT', 'ResNet50', 'ResNet101', 'ResNet152', 'Mobilenet_v2'), on_change=onChangeModel)
    num_output = st.sidebar.number_input("Select number of output images", 1, num_database_images, 1, 1)

    # Initialize the model, preprocess function, and dimension if not already done
    if st.session_state['model'] is None:
        with st.spinner('Initializing model...'):
            st.session_state['model'], st.session_state['preprocess'], st.session_state['dimension'] = init_model(model_name, device)

    # Build the faiss index if not already done
    if st.session_state['fs_idx'] is None:
        progress_bar = st.progress(0, text="Building faiss")
        st.session_state['fs_idx'], st.session_state['img_idx'] = build_faiss(progress_bar, st.session_state['db_path'], st.session_state['model'], st.session_state['preprocess'], st.session_state['dimension'], device)
        progress_bar.empty()

    # Upload an image and perform similarity search
    input_cols = st.sidebar.columns(2)
    uploaded_file = input_cols[0].file_uploader("Search by Image: ")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(BytesIO(bytes_data))
        st.sidebar.image(img, caption="Query image")

        # Compute the embedding of the query image
        embedding = return_image_embedding(img, st.session_state['model'], st.session_state['preprocess'], device)
        embedding = np.array(dtype=np.float32)
        # Perform similarity search using the faiss index
        D, I = st.session_state['fs_idx'].search(embedding, num_output)

        # Display the similar images
        columns = st.columns(3)
        for n, index_image in enumerate(I[0]):
            img_name = st.session_state['img_idx'][index_image]
            image = Image.open(os.path.join(st.session_state['db_path'], img_name))
            columns[n % 3].image(image, caption=f'{img_name}')

    # Perform similarity search by text
    search_key = input_cols[1].text_input("Search by key")
    if search_key != "":
        # Compute the embedding of the query text
        embedding = return_text_embedding(search_key, st.session_state['clip_model'], device)
        embedding = np.array(dtype=np.float32)
        # Perform similarity search using the faiss index
        D, I = st.session_state['fs_idx_clip'].search(embedding, num_output)

        # Display the similar images
        columns = st.columns(3)
        for n, index_image in enumerate(I[0]):
            img_name = st.session_state['img_idx_clip'][index_image]
            image = Image.open(os.path.join(st.session_state['db_path'], img_name))
            columns[n % 3].image(image, caption=f'{img_name}')