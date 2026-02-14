from .lenet import build_lenet
from .vgg16 import build_vgg16
from .resnet18 import build_resnet18
from .resnet50 import build_resnet50
from .mobilenetv2 import build_mobilenetv2
from .efficientnetb0 import build_efficientnetb0
from .convnext_tiny import build_convnext_tiny


def build_model(model_name, num_classes):
    if model_name == "lenet":
        return build_lenet(num_classes)
    elif model_name == "vgg16":
        return build_vgg16(num_classes)
    elif model_name == "resnet18":
        return build_resnet18(num_classes)
    elif model_name == "resnet50":
        return build_resnet50(num_classes)
    elif model_name == "mobilenetv2":
        return build_mobilenetv2(num_classes)
    elif model_name == "efficientnetb0":
        return build_efficientnetb0(num_classes)
    elif model_name == "convnext_tiny":
        return build_convnext_tiny(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
