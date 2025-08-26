import vllm
from pathlib import Path
QWEN25VL7B = "/home6/fzy/models/Qwen2.5-VL-7B-Instruct"


def get_model_name(model_name_or_path):
    t = model_name_or_path.split("/")[-1]
    t = t.replace('-', '_')
    return t

def label_ocrv2(model_name_or_path: str):
    # 1. load vllm model using all the GPUs
    
    
    # 2. do inference 


    # 3. store the result into json, sample item as follows and store the images
    # {
    #     "id": "0054c91397f2fe05",
    #     "conversations": [
    #         {
    #             "from": "human",
    #             "value": "what is the brand of phone?\n<image>"
    #         },
    #         {
    #             "from": "gpt",
    #             "value": "nokia"
    #         }
    #     ],
    #     "image_abs": "/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/dataset/Images/textvqa/train/images/0054c91397f2fe05.jpg",
    #     "image": "images/0054c91397f2fe05.jpg"
    # },
    pass

if __name__ == '__main__':
    pass