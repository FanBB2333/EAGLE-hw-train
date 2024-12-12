import os

AVAILABLE_MODELS = {
    "eagle": "Eagle",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError:
        pass


# BEGIN hxl
# import hf_transfer

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# END hxl
