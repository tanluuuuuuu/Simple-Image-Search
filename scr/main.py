import streamlit as st
from PIL import Image
from io import BytesIO
import os
import torch
import tkinter as tk
from tkinter import filedialog

from utils import *
from model_embedding import *

# Initialize session states
def init_states():
    if 'faiss_index' not in st.session_state:
        st.session_state['faiss_index'] = None
        st.session_state['images_index'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
        st.session_state['preprocess'] = None
        st.session_state['dimension'] = None
    if 'database_path' not in st.session_state:
        st.session_state['database_path'] = None

# Reset model-related session states
def onChangeModel():
    st.session_state['model'] = None
    st.session_state['preprocess'] = None
    st.session_state['dimension'] = None
    st.session_state['faiss_index'] = None  
    st.session_state['images_index'] = None
    
def onChangeDevice():
    st.empty()
    pass

if __name__ == '__main__':
    # Initialize session states and create sidebar title
    init_states()
    
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)
    
    st.sidebar.title("Image Search")

    # Choose device (CPU or CUDA)
    device = st.sidebar.selectbox('Choose device', ('cpu', 'cuda'), on_change=onChangeDevice)
    if device == 'cuda' and not torch.cuda.is_available():
        st.warning("CUDA is not available")
        st.stop()

    # Set the path to the image database
    st.sidebar.write('Please select a folder:')
    col_folder = st.sidebar.columns(2)
    clicked = col_folder[0].button('Folder Picker')
    clicked2 = col_folder[1].button('Use sample database')
    if clicked:
        st.session_state['database_path'] = st.sidebar.text_input('Selected folder database:', filedialog.askdirectory(master=root))
    elif clicked2:
        current_directory = os.path.abspath(os.getcwd())
        st.session_state['database_path'] = os.path.join(current_directory, 'sample_data')
        st.session_state['database_path'] = st.sidebar.text_input('Selected folder database:', value=st.session_state['database_path'])
    elif st.session_state['database_path'] == None:
        st.sidebar.write("Select database path to continue")
        st.stop()
    num_database_images = checkDatabasePath(st.session_state['database_path'])
    st.sidebar.success(f"Found {num_database_images} images in database")

    # Choose the model and number of output images
    model_name = st.sidebar.selectbox('Choose model', ('ViT', 'ResNet50', 'ResNet101', 'ResNet152', 'Mobilenet_v2'), on_change=onChangeModel)
    num_output = st.sidebar.number_input("Select number of output images", 1, num_database_images, 5, 1)

    # Initialize the model, preprocess function, and dimension if not already done
    if st.session_state['model'] is None:
        with st.spinner('Initializing model...'):
            st.session_state['model'], st.session_state['preprocess'], st.session_state['dimension'] = init_model(model_name, device)

    # Build the faiss index if not already done
    if st.session_state['faiss_index'] is None:
        progress_bar = st.progress(0, text="Building faiss")
        st.session_state['faiss_index'], st.session_state['images_index'] = build_faiss(progress_bar, st.session_state['database_path'], st.session_state['model'], st.session_state['preprocess'], st.session_state['dimension'], device)
        progress_bar.empty()

    # Upload an image and perform similarity search
    uploaded_file = st.sidebar.file_uploader("Upload an Image: ")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(BytesIO(bytes_data))
        st.sidebar.image(img, caption="Query image")

        # Compute the embedding of the query image
        embedding = return_image_embedding(img, st.session_state['model'], st.session_state['preprocess'], device)

        # Perform similarity search using the faiss index
        D, I = st.session_state['faiss_index'].search(embedding, num_output)

        # Display the similar images
        columns = st.columns(3)
        for n, index_image in enumerate(I[0]):
            img_name = st.session_state['images_index'][index_image]
            image = Image.open(os.path.join(st.session_state['database_path'], img_name))
            columns[n % 3].image(image, caption=f'{img_name}')
