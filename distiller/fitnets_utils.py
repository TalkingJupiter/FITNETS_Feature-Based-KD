import torch.nn as nn

def get_layer_by_name(model, layer_name):
    """
    Recursively find a layer in a model by name.
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"layer '{layer_name}' not found.")

def register_feature_hook(model, layer_name, feature_storage):
    """
    Attach a forward hook to a layer to store its output.
    """
    layer = get_layer_by_name(model, layer_name)
    def hook_fn(_, __, output):
        feature_storage['feature'] = output
    layer.register_forward_hook(hook_fn)
    