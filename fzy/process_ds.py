import json
import random
random.seed(42)
import jsonlines
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
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
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "Charades":
        # duration = data["answer"][1] - data["answer"][0]
        # duration = float(f"{duration:.2f}")
        duration = data["duration"]
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "QVHighlights":
        duration = data["duration"]
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


def ds2json(ds, cut_videos=False):
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
        # question = item['question']
        if ds.name in ["ActivityNetCaps", "Charades", "QVHighlights", "valor", "youcook2"]:
            video_path = item['data_path']
            answer = f"{item['answer'][0]}s"
            conv, question1, question2 = gen_prompt(item, ds.name)
        elif ds.name == "ActivityNetQA":
            question1 = item['question']
            answer = item['answer']
            video_path = item['data_path']
        else:
            raise NotImplementedError(f"Dataset {ds.name} not implemented")
        # duration = item['duration']
        # get result from gen_prompt
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
    def cut():
        process_videos(
            input_folder=str(ds.video_path),
            output_folder=str(output_dir_ds / "videos")
        )
        print(f"Processed videos for {ds.name} dataset, saved to {output_dir_ds / 'videos'}")
    if cut_videos:
        cut()
    return ret

def process_ds():
    # ActivityNet, Breakfast, Charades, QVHighlights, VALOR32K, YouCook2
    # 实例化
    # anc = ActivityNetCaps(train=True)
    # charades = Charades(train=True)
    # qvhl = QVHighlights(train=True)
    anqa = ActivityNetQA(train=True)
    # YouCook2()
    print(f"Loading ds done")
    
    # initialize anc
    # anc_obj = ds2json(anc)
    # charades_obj = ds2json(charades)
    # qvhl_obj = ds2json(qvhl)
    # anqa_obj = ds2json(anqa)
    # return [anc_obj, charades_obj, qvhl_obj, anqa_obj]



def combine_ds():
    # 1.combine all the datasets into one json file
    
    # 2. copy video files to the combined videos folder
    output_dir = OUTPUT_DIR / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_data = []
    for ds_name in ["ActivityNetCaps", "Charades", "QVHighlights", "ActivityNetQA"]:
        ds_file = OUTPUT_DIR / ds_name / f"{ds_name}.json"
        with open(ds_file, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)
    # re-assign ids
    for idx, item in enumerate(combined_data):
        item['id'] = str(idx)
    # save combined data
    combined_file = output_dir / "combined.json"
    combined_video_dir = output_dir / "videos"
    combined_video_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    print(f"Combined dataset saved to {combined_file}")
    # copy videos
    for ds_name in ["ActivityNetCaps", "Charades", "QVHighlights", "ActivityNetQA"]:
        ds_video_dir = OUTPUT_DIR / ds_name / "videos"
        # copy all the files from ds_video_dir to combined_video_dir, not only mp4
        for video_file in ds_video_dir.glob("*"):
            if video_file.is_file():
                new_video_path = combined_video_dir / video_file.name
                if not new_video_path.exists():
                    video_file.rename(new_video_path)
    print(f"Copied videos to {combined_video_dir}")
    return combined_data
    

def is_valid_video(video_path):
    try:
        if not os.path.exists(video_path):
            return False

        cv2_vr = cv2.VideoCapture(video_path)
        if not cv2_vr.isOpened():
            return False

        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        if duration <= 0:
            cv2_vr.release()
            return False

        frame_id_list = np.linspace(0, duration - 1, 8, dtype=int)

        for frame_idx in frame_id_list:
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret or frame is None:
                cv2_vr.release()
                return False
            _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2_vr.release()
        return True

    except Exception as e:
        return False

def check_all_videos(root_dir, save_invalid_path="invalid_videos.txt"):
    invalid_videos = []
    total = 0
    checked = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mp4"):
                total += 1
                video_path = os.path.join(dirpath, filename)
                if not is_valid_video(video_path):
                    invalid_videos.append(video_path)
                    print(f"❌ 无效: {video_path}")
                else:
                    print(f"✅ 有效: {video_path}")
                checked += 1

    # 保存无效视频路径到文件
    # with open(save_invalid_path, "w") as f:
    #     for path in invalid_videos:
    #         f.write(path + "\n")

    print("\n✅ 检查完毕")
    print(f"总视频数量: {total}")
    print(f"检测视频数量: {checked}")
    print(f"无效视频数量: {len(invalid_videos)}")
    # print(f"无效视频已保存到: {save_invalid_path}")



if __name__ == "__main__":
    process_ds()
    # combine_ds()
    # check_all_videos("/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/added/processed/combined/videos")
    
    
    # tds = TextVQA()
    # tds.offload_train()
    # doc = DocVQA()
    # doc.reformat()
    
    # anqa = ActivityNetQA()
    # anqa.reformat()
    