import torch
import torchvision
from torchvision.io import read_image
from torchvision.models import vit_b_16, resnet50, resnet101, resnet152, mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms.functional as TF

def init_model(name, device):
    # Define the available model options and their corresponding weights
    model_options = {
        'ViT': (vit_b_16, 'ViT_B_16_Weights', 768),
        'ResNet50': (resnet50, 'ResNet50_Weights', 2048),
        'ResNet101': (resnet101, 'ResNet101_Weights', 2048),
        'ResNet152': (resnet152, 'ResNet152_Weights', 2048),
        'Mobilenet_v2': (mobilenet_v2, 'MobileNet_V2_Weights', 1280)
    }
    
    # Retrieve the model class, weights, and embedding dimension based on the selected name
    model_class, weights, dimension = model_options[name]
    weights = getattr(torchvision.models, weights).DEFAULT
    
    # Load the pre-trained model and set it to evaluation mode
    model = model_class(weights=weights)
    model = create_feature_extractor(model, 
                                     return_nodes={"flatten": "output"} if name != 'ViT' else {"getitem_5": "output"})
    model.eval().to(device)
    
    preprocess = weights.transforms()
    
    return model, preprocess, dimension

def return_image_embedding(img, model, preprocess, device):
    # Convert the image to a tensor and move it to the specified device
    tensor_img = TF.to_tensor(img).to(device)
    
    # Preprocess the tensor image and generate the embedding
    batch = preprocess(tensor_img).unsqueeze(0)
    embedding = model(batch)['output'].detach().cpu()
    
    return embedding
