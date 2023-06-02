import streamlit as st
from PIL import Image
from io import BytesIO
import os
import torch

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

# Reset model-related session states
def onChangeModel():
    st.session_state['model'] = None
    st.session_state['preprocess'] = None
    st.session_state['dimension'] = None

if __name__ == '__main__':
    # Initialize session states and create sidebar title
    init_states()
    st.sidebar.title("Image Search")

    # Choose device (CPU or CUDA)
    device = st.sidebar.selectbox('Choose device', ('cpu', 'cuda'))
    if device == 'cuda' and not torch.cuda.is_available():
        st.warning("CUDA is not available")
        st.stop()

    # Set the path to the image database
    database_path = "D:/63. CinnamonAI entrance test/Image Search/data/smaller_database"

    # Choose the model and number of output images
    model_name = st.sidebar.selectbox('Choose model', ('ViT', 'ResNet50', 'ResNet101', 'ResNet152', 'Mobilenet_v2'), on_change=onChangeModel)
    num_output = st.sidebar.number_input("Select number of output images", 0, len(os.listdir(database_path)), 5, 1)

    # Initialize the model, preprocess function, and dimension if not already done
    if st.session_state['model'] is None:
        with st.spinner('Initializing model...'):
            st.session_state['model'], st.session_state['preprocess'], st.session_state['dimension'] = init_model(model_name, device)

    # Build the faiss index if not already done
    if st.session_state['faiss_index'] is None:
        progress_bar = st.progress(0, text="Building faiss")
        st.session_state['faiss_index'], st.session_state['images_index'] = build_faiss(progress_bar, database_path, st.session_state['model'], st.session_state['preprocess'], st.session_state['dimension'], device)
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
            image = Image.open(os.path.join(database_path, img_name))
            columns[n % 3].image(image, caption=f'{img_name}')
