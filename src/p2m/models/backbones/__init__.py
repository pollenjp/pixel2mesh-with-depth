# First Party Library
from p2m.models.backbones.resnet import resnet50
from p2m.models.backbones.vgg16 import VGG16P2M
from p2m.models.backbones.vgg16 import VGG16Recons
from p2m.models.backbones.vgg16 import VGG16TensorflowAlign
from p2m.options import ModelBackbone


def get_backbone(options):
    match options.backbone:
        case ModelBackbone.VGG16:
            if options.align_with_tensorflow:
                nn_encoder = VGG16TensorflowAlign()
            else:
                nn_encoder = VGG16P2M(pretrained="pretrained" in options.backbone.name)
            nn_decoder = VGG16Recons()
        case ModelBackbone.RESNET50:
            nn_encoder = resnet50()
            nn_decoder = None
        case _:
            raise NotImplementedError(f"No implemented backbone called '{options.backbone.name}' found")
    return nn_encoder, nn_decoder
