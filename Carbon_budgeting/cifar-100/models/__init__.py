"""
models/__init__.py  –  CIFAR-100 model registry

Usage in the notebook:
    from models import get_model, preprocess_data

    x_train_m, x_test_m = preprocess_data(MODEL_NAME, x_train, x_test)
    model = get_model(MODEL_NAME, num_classes=100)
"""

from .lenet5         import get_lenet5,         preprocess as _pre_lenet5
from .vgg16          import get_vgg16,          preprocess as _pre_vgg16
from .resnet18       import get_resnet18,       preprocess as _pre_resnet18
from .resnet50       import get_resnet50,       preprocess as _pre_resnet50
from .mobilenetv2    import get_mobilenetv2,    preprocess as _pre_mobilenetv2
from .efficientnetb0 import get_efficientnetb0, preprocess as _pre_efficientnetb0

_MODEL_REGISTRY = {
    "lenet5":          (get_lenet5,         _pre_lenet5),
    "vgg16":           (get_vgg16,          _pre_vgg16),
    "resnet18":        (get_resnet18,       _pre_resnet18),
    "resnet50":        (get_resnet50,       _pre_resnet50),
    "mobilenetv2":     (get_mobilenetv2,    _pre_mobilenetv2),
    "efficientnetb0":  (get_efficientnetb0, _pre_efficientnetb0),
}


def get_model(model_name: str, num_classes: int = 100):
    """Return a compiled Keras model for the given model_name."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY)}"
        )
    builder, _ = _MODEL_REGISTRY[model_name]
    return builder(num_classes)


def preprocess_data(model_name: str, x_train, x_test):
    """
    Apply the correct preprocessing for the given model.
    Input x_train / x_test must already be in [0, 1] float range
    (standard CIFAR normalization).
    Returns (x_train_processed, x_test_processed).
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY)}"
        )
    _, prep_fn = _MODEL_REGISTRY[model_name]
    return prep_fn(x_train, x_test)
