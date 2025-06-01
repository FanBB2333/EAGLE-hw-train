import json
import random
random.seed(42)
import jsonlines
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms as transforms
import cv2
from pathlib import Path
try:
    from eagle.model.builder import load_pretrained_model
    from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from eagle.conversation import conv_templates, SeparatorStyle
except ImportError:
    raise ImportError("Import Error")
from fzy.ds import *
from fzy.video_8_frame_opencv import process_videos
OUTPUT_DIR = DATASET_BASE / 'processed'

def gen_prompt(data, task):
    question2 = None
    if task == "ActivityNetCaps":
        duration = data["duration"]
        # question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
        # question2 = f'The video\'s duration is {duration}s. The event "{data["question"]}" starts at: '
        # question2 = "The event starts at 00:"
        # question1 = "What is the video about?"
        # question2 = "The video is about: "
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "Charades":
        duration = data["answer"][1] - data["answer"][0]
        duration = float(f"{duration:.2f}")
        # question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video. The event starts at:'
        # question2 = f'The video\'s duration is {duration}s. The event "{data["question"]}" starts at: '
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "QVHighlights":
        duration = data["duration"]
        answers = data["answer"]
        # question1 = f"{data['question']}"
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task in ["valor", "youcook2"]:
        duration = data["duration"]
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    else:
        raise NotImplementedError(f"Task {task} not implemented")

    if DEFAULT_IMAGE_TOKEN not in question1:
        question1 = DEFAULT_IMAGE_TOKEN + '\n' + question1
    # args.conv_template: llama3
    conv = conv_templates["llama3"].copy()
    # 0: user, 1: assistant
    conv.append_message(conv.roles[0], question1)
    if question2 is not None:
        conv.append_message(conv.roles[1], question2)
    prompt_question = conv.get_prompt()
    return conv, question1, question2


# target data format:
sample_format_lambda = lambda question, answer, image_path, id: {
    'id': str(id),
    'conversations': [
        {'from': 'human', 'value': question},
        {'from': 'gpt', 'value': answer}
    ],
    'image': image_path
    # In [4]: data[0]
    # Out[4]: 
    # {'id': '0',
    # 'conversations': [{'from': 'human',
    # 'value': 'Write a terse but informative summary of the following video clip.\n<image>'},
    # {'from': 'gpt',
    # 'value': 'Oaxaca de juarez, mexico - circa 1970: mexican tourists on the square of the cathedral of our lady of the assumption in the city of oaxaca. archival of mexico in oaxaca state in the 1970s.'}],
    # 'image': 'valley/076101_076150/1043215450.mp4'}

    # In [5]: len(data)
    # Out[5]: 702360
}


def ds2json(ds):
    ret = list()
    for idx, item in enumerate(tqdm(ds)):
    # Sample original data format:
    # ({
    #     'data_path': str(video_file),
    #     'question': sentence,
    #     'answer': timestamps,
    #     'duration': v['duration'],
    # })
        # print(item)
        video_path = item['data_path']
        # question = item['question']
        answer = f"{item['answer'][0]}s"
        # duration = item['duration']
        # get result from gen_prompt
        conv, question1, question2 = gen_prompt(item, ds.name)
        ds_item = sample_format_lambda(
            question=question1,
            answer=answer,
            image_path=video_path,
            id=idx
        )
        ret.append(ds_item)
    # save to json file
    output_dir_ds = OUTPUT_DIR / f"{ds.name}"
    output_dir_ds.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_ds / f"{ds.name}.json"
    with open(output_file, 'w') as f:
        json.dump(ret, f, indent=4)
    print(f"Processed {ds.name} dataset, saved to {output_file}")
    # save processed video files
    process_videos(
        input_folder=str(ds.video_path),
        output_folder=str(output_dir_ds / "videos")
    )
    print(f"Processed videos for {ds.name} dataset, saved to {output_dir_ds / 'videos'}")
    return True


def process_ds():
    # ActivityNet, Breakfast, Charades, QVHighlights, VALOR32K, YouCook2
    # 实例化
    anc = ActivityNetCaps(train=True)
    charades = Charades(train=True)
    qvhl = QVHighlights(train=True)
    # YouCook2()
    print(f"Loading ds done")
    
    # initialize anc
    # anc_obj = ds2json(anc)
    charades_obj = ds2json(charades)
    qvhl_obj = ds2json(qvhl)



if __name__ == "__main__":
    process_ds()