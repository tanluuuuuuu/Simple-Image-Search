
# Simple Image Search

The Image Search App is a powerful tool that allows users to search for specific images or find visually similar images within a database. It utilizes advanced computer vision techniques to extract features from images and perform similarity matching, enabling various applications such as searching for faces or finding similar products using images.

## Features
- Image Search: Users can search for specific images by uploading an image or providing a query description. The app matches the query against the image database and retrieves relevant results based on image similarity.
- Similar Image Retrieval: Users can find visually similar images to a given query image within the database. The app employs advanced image embedding techniques and similarity algorithms to identify images with comparable visual characteristics.

## Installation
1. Clone the repository: git clone https://github.com/tanluuuuuuu/Simple-Image-Search.git
2. Navigate to the project directory: cd Simple-Image-Search
3. Install the required dependencies: pip install -r requirements.txt

## Usage
1. Start the application: python app.py
2. Open your web browser and go to http://localhost:5000
3. Upload an image or provide a query description to search for similar images.
4. View the search results and explore visually similar images.

## Technologies Used
- Programming Language: Python
- Computer Vision Libraries: Pillow, PyTorch
- Deployment: Streamlit 

## Docker
You can also run the Image Search App using Docker. Follow the steps below:
1. Install Docker on your machine.
2. Build the Docker image:
```shell
docker build -t image-search-app .
```
3. Run the Docker container:
```shell
docker run -p 8051:8051 image-search-app
```
4. Open your web browser and go to http://localhost:8051
5. Provide database path then upload an image to search for similar images.
6. View the search results and explore visually similar images.

## Contributing
Contributions to the Image Search App are welcome! If you find any bugs or have suggestions for new features, please open an issue or submit a pull request. Follow the guidelines in the CONTRIBUTING.md file for contributing to the project.

## Contact
For any inquiries or feedback, please contact [luu432002@gmail.com]

## Acknowledgments
We would like to thank the contributors and the open-source community for their support and contributions to the Image Search App.

**Note**: The Image Search App is for demonstration purposes only and may require additional configuration or customization to suit specific use cases or production environments.