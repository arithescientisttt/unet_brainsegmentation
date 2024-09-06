dependencies = ["torch"]

import torch
from unet_model import UNet

def build_unet(pretrained=False, **kwargs):
    """
    Constructs the U-Net model for biomedical image segmentation with optional pretrained weights.
    
    Args:
        pretrained (bool): If True, loads pretrained weights into the model.
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
        init_features (int): Initial number of feature maps in the first encoder layer.
    
    Returns:
        model (UNet): A U-Net model instance.
    """
    model = UNet(**kwargs)

    if pretrained:
        weights_url = "https://github.com/arithescientisttt/unet_brainsegmentation/weights/unet.pt"
        pretrained_weights = torch.hub.load_state_dict_from_url(weights_url, progress=False, map_location='cpu')
        model.load_state_dict(pretrained_weights)

    return model
