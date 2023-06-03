from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from torchvision.models import alexnet, AlexNet_Weights, mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torchvision.transforms.functional as transform

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).to("cuda")
train_nodes, eval_nodes = get_graph_node_names(mobilenet_v2())
print(model)
print(eval_nodes[-5:])
print(train_nodes[-5:])

return_nodes = {
    "flatten": "output"
}
model.eval()
model = create_feature_extractor(model, return_nodes=return_nodes)
preprocess = weights.transforms()