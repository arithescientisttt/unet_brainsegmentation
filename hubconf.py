dependencies = ["torch",'torchvision']

import torch

from unet import UNet


def unet(pretrained=False, **kwargs):
    """
    U-Net segmentation model with batch normalization for biomedical image segmentation
    pretrained (bool): load pretrained weights into the model
    in_channels (int): number of input channels
    out_channels (int): number of output channels
    init_features (int): number of feature-maps in the first encoder layer
    """
    model = UNet(**kwargs)

    # Load pre-trained weights
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            'https://raw.githubusercontent.com/arithescientisttt/unet_briansegmentation/main/weights/unet.pt',
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
    
    return model