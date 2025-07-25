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
from datasets import load_dataset
import socket
hostname = socket.gethostname()
PROJECT_BASE = Path(__file__).resolve().parents[1]
if hostname == '9008':
    DATASET_BASE = PROJECT_BASE / 'dataset'
elif hostname == '9007': # hxl-246
    DATASET_BASE = PROJECT_BASE / 'dataset/Video/added'
    # DATASET_BASE = PROJECT_BASE / 'dataset/Video/added'
else:
    DATASET_BASE = PROJECT_BASE / 'dataset'
    
DEFAULT_VIDEO_TOKEN = '<video_token>'
def get_video_length(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return None

    # 获取总帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算视频时长（秒）
    if fps > 0:
        video_length = frame_count / fps
    else:
        print("Unable to retrieve FPS from the video.")
        return None

    # 释放视频资源
    cap.release()

    return video_length

form_yt_url = lambda x: f"https://www.youtube.com/watch?v={x}"
def get_vid(fullname):
    # x-2Abohj8VY_30.000_40.000 -> x-2Abohj8VY
    # LXI2eW_dZoU_30.000_40.000 -> LXI2eW_dZoU
    # split the video name
    # 找到倒数第二个下划线的位置
    second_last_underscore = fullname.rfind('_', 0, fullname.rfind('_'))
    
    return fullname[:second_last_underscore]


# target data format:
sample_format_lambda = lambda question, answer, image_path, id, path_prefix="videos": {
    'id': str(id),
    'conversations': [
        {'from': 'human', 'value': question},
        {'from': 'gpt', 'value': answer}
    ],
    'image_abs': image_path, # /home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/added/ActivityNetCaps/v1-3/train_val/v_ehGHCYKzyZ8.mp4
    'image': f"{path_prefix}/{image_path.split('/')[-1]}" # v_ehGHCYKzyZ8.mp4
}
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


class MMDS(Dataset):
    def __init__(self, name):
        self.name = name
        self.data = list()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ActivityNetCaps(MMDS):
    def __init__(self, db_path = DATASET_BASE / 'ActivityNetCaps', train=False):
        name = 'ActivityNetCaps'
        super().__init__(name)
        self.db_path = db_path
        self.train = json.load(open(db_path / 'train.json'))
        self.val1 = json.load(open(db_path / 'val_1.json'))
        self.val2 = json.load(open(db_path / 'val_2.json'))
        # self.video_path = self.db_path / "all_test"
        self.video_path = self.db_path / "v1-3/train_val"
        print(f"[{name}] length of val1: {len(self.val1)}, val2: {len(self.val2)}")
        if not train:
            self.load_data()
        else:
            self.load_train()
    
    def load_data(self):
        # load from the val1 and val2 data
        splits = [self.val1, self.val2, self.train]
        split = splits[0]
        self.data = list()
        for k, v in split.items():
            # k: id
            # v: {'duration': 55.15, 'timestamps': [[0.28, 55.15], [13.79, 54.32]], 'sentences': ['A weight lifting tutorial is given.', '  The coach helps the guy in red with the proper body placement and lifting technique.']}
            video_file = self.video_path / f"{k}.mp4"
            if not os.path.exists(video_file):
                video_file = video_file.with_suffix(".mkv")
                if not os.path.exists(video_file):
                    # print(f"Video file not found: {video_file}")
                    continue
            for idx in range(len(v['sentences'])):
                sentence = v['sentences'][idx]
                timestamps = v['timestamps'][idx]
                self.data.append({
                    'data_path': str(video_file),
                    'question': sentence,
                    'answer': timestamps,
                    'duration': v['duration'],
                })

        print(f"[{self.name}] length of data: {len(self.data)}")
    
    def load_train(self):
        # load from the val1 and val2 data
        splits = [self.val1, self.val2, self.train]
        split = splits[2]
        self.data = list()
        for k, v in split.items():
            # k: id
            # v: {'duration': 55.15, 'timestamps': [[0.28, 55.15], [13.79, 54.32]], 'sentences': ['A weight lifting tutorial is given.', '  The coach helps the guy in red with the proper body placement and lifting technique.']}
            video_file = self.video_path / f"{k}.mp4"
            if not os.path.exists(video_file):
                video_file = video_file.with_suffix(".mkv")
                if not os.path.exists(video_file):
                    # print(f"Video file not found: {video_file}")
                    continue
            for idx in range(len(v['sentences'])):
                sentence = v['sentences'][idx]
                timestamps = v['timestamps'][idx]
                self.data.append({
                    'data_path': str(video_file),
                    'question': sentence,
                    'answer': timestamps,
                    'duration': v['duration'],
                })

        print(f"[Train] [{self.name}] length of data: {len(self.data)}")
        

class Breakfast(MMDS):
    def __init__(self, db_path = DATASET_BASE / 'breakfast'):
        name = 'Breakfast'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path / "segmentation_coarse"
        self.video_path = self.db_path / "BreakfastII_15fps_qvga_sync"
        self.load_data()
        
    def load_data(self):
        # list the subdirs in video_path with depth=1
        dir_p = [x for x in self.video_path.iterdir() if x.is_dir()]
        # iter over the dirs
        data = list()
        for p in tqdm(dir_p, desc="Loading Breakfast data"):
            dir_cams = [x for x in p.iterdir() if x.is_dir()]
            for cam in dir_cams:
                video_files = list(cam.glob('*.avi'))
                for video_file in video_files:
                    video_id = video_file.stem
                    anno_file = cam / f"{video_id}.avi.labels"
                    if not anno_file.exists():
                        # print(f"Annotation file not found: {anno_file}, video: {video_file}")
                        continue
                    video_seg = list()
                    with open(anno_file, "r") as f:
                        lines = f.readlines()
                    # line sample: 55-233 pour_cereals 
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 2:
                            continue
                        start, end = parts[0].split("-")
                        start = int(start)
                        end = int(end)
                        action = parts[1]
                        video_seg.append({
                            "start": start,
                            "end": end,
                            "action": action,
                        })
                    data.append({
                        "data_path": str(video_file),
                        "segments": video_seg,
                        'duration': get_video_length(video_file),
                    })
        self.data = data
        print(f"[{self.name}] length of data: {len(self.data)}")
    

class Charades(MMDS):
    def __init__(self, db_path = DATASET_BASE / 'Charades', train=False):
        name = 'Charades'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path
        self.video_path = self.db_path / "Charades_v1_480"
        if not train:
            self.load_data()
        else:
            self.load_train()
    
    def load_data(self):
        test_name = "charades_sta_test.txt"
        # each line is a sample
        with open(self.anno_path / test_name, "r") as f:
            lines = f.readlines()
        # line sample: 3MSZA 24.3 30.4##person turn a light on.
        data = list()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split("##")
                if len(parts) != 2:
                    raise ValueError("Invalid format in line: " + line)

                video_info, description = parts
                video_id, start_time, end_time = video_info.split()
                video_file = self.video_path / f"{video_id}.mp4"
                if not video_file.exists():
                    continue
                duration = get_video_length(video_file)
                if duration is None:
                    continue

                start_time = float(start_time)
                end_time = float(end_time)
                sample = {
                    "data_path": str(video_file),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": description,
                    "duration": duration,
                }
                data.append(sample)
            except Exception as e:
                print(f"Failed to parse line: {line}, error: {e}")
        print(f"Loaded {len(data)} samples ")
    
        # convert to "data_path": ,question: ,answer:
        ret = list()
        for item in data:
            query = item["description"]
            duration = item["end_time"] - item["start_time"]
            ret.append({
                "data_path": item["data_path"],
                "question": query,
                "answer": [item["start_time"], item["end_time"]],
                'duration': item['duration'],
            })
        self.data = ret
                
    def collate_fn(self, batch):
        return batch
    
    
    def load_train(self):
        test_name = "charades_sta_train.txt"
        # each line is a sample
        with open(self.anno_path / test_name, "r") as f:
            lines = f.readlines()
        # line sample: 3MSZA 24.3 30.4##person turn a light on.
        data = list()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split("##")
                if len(parts) != 2:
                    raise ValueError("Invalid format in line: " + line)

                video_info, description = parts
                video_id, start_time, end_time = video_info.split()
                video_file = self.video_path / f"{video_id}.mp4"
                if not video_file.exists():
                    continue
                start_time = float(start_time)
                end_time = float(end_time)
                duration = get_video_length(video_file)
                if duration is None:
                    continue
                sample = {
                    "data_path": str(video_file),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": description,
                    "duration": duration,
                }
                data.append(sample)
            except Exception as e:
                print(f"Failed to parse line: {line}, error: {e}")
    
        # convert to "data_path": ,question: ,answer:
        ret = list()
        for item in data:
            query = item["description"]
            duration = item["end_time"] - item["start_time"]
            ret.append({
                "data_path": item["data_path"],
                "question": query,
                "answer": [item["start_time"], item["end_time"]],
                'duration': item['duration'],
            })
        self.data = ret
        print(f"[Train] [{self.name}] Loaded {len(data)} samples ")



class QVHighlights(MMDS):
    def __init__(self, db_path = DATASET_BASE / 'QVHighlights', train=False):
        name = 'QVHighlights'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path / 'annotations'
        self.video_path = self.db_path / 'videos'
        self.test = self.load_jsonl(self.anno_path / 'highlight_test_release.jsonl')
        self.train = self.load_jsonl(self.anno_path / 'highlight_train_release.jsonl')
        self.val = self.load_jsonl(self.anno_path / 'highlight_val_release.jsonl')
        if not train:
            self.load_data()
        else:
            self.load_train()
    
    
    def load_jsonl(self, path):
        data = list()
        with jsonlines.open(path) as reader:
            for obj in reader:
                data.append(obj)
        return data
    def load_data(self):
        splits = {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }
        data = list()
        for line in self.test:
            video_file = self.video_path / f"{line['vid']}.mp4"
            if not video_file.exists():
                continue
            data.append({
                'data_path': str(video_file),
                'question': "What does the video show?",
                'answer': line['query'],
                'duration': line['duration'],
                'qid': line['qid'],
            })
        splits['test'] = deepcopy(data)
        
        data = list()
        for line in self.val:
            video_file = self.video_path / f"{line['vid']}.mp4"
            if not video_file.exists():
                continue
            data.append({
                'data_path': str(video_file),
                'question': line['query'],
                'answer': line['relevant_windows'],
                'duration': line['duration'],
                'qid': line['qid'],
            })
        splits['val'] = deepcopy(data)
        print(f"[{self.name}] length of val data: {len(splits['val'])}, test: {len(splits['test'])}")
        self.data = splits['val']
        

    def load_train(self):
        splits = {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }
        data = list()
        for line in self.train:
            video_file = self.video_path / f"{line['vid']}.mp4"
            if not video_file.exists():
                continue
            data.append({
                'data_path': str(video_file),
                'question': line['query'],
                'answer': line['relevant_windows'][0],
                'duration': line['duration'],
                'qid': line['qid'],
            })
        splits['train'] = deepcopy(data)
        
        print(f"[Train] [{self.name}] length of train data: {len(splits['train'])}")
        self.data = splits['train']


class VALOR32K(MMDS):
    def __init__(self, db_path = DATASET_BASE / 'valor32k'):
        name = 'valor32k'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path
        self.video_path = self.db_path / "videos"
        self.load_data()

    def sample(self, n=18):
        # load the json file
        data = json.load(open(self.anno_path / "desc_test.json"))
        print(f"[{self.name}] Total {len(data)} samples, sampling {n} samples")
        # random sample n samples
        sampled = random.sample(data, n)
        sampled = deepcopy(sampled)
        urls = list()
        for item in data:
            vid = item['video_id']
            urls.append(form_yt_url(get_vid(vid)))
        return urls
    
    def load_data(self):
        desc_test = json.load(open(self.anno_path / "desc_test.json"))
        id2test = dict()
        for test_data in desc_test:
            vid_full = test_data['video_id']
            vid = get_vid(vid_full)
            if vid not in id2test:
                id2test[vid] = list()
            id2test[vid].append(test_data)
            
        # list the videos in the video_path
        video_files = list(self.video_path.glob("*.mp4"))
        data = list()
        for video_file in video_files:
            video_id = video_file.stem
            duration = get_video_length(video_file)
            if video_id not in id2test:
                continue
            for test_data in id2test[video_id]:
                query = test_data['desc']
                time_splits = test_data['video_id'].split("_")
                start_time = float(time_splits[-2])
                end_time = float(time_splits[-1])
                data.append({
                    'data_path': str(video_file),
                    'question': query,
                    'duration': duration,
                    'answer': [start_time, end_time],
                })
        self.data = data
        print(f"[{self.name}] length of data: {len(self.data)}")



class YouCook2(MMDS):
    def __init__(self, db_path = DATASET_BASE / 'youcook2', train=False):
        name = 'youcook2'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path
        self.video_path = self.db_path / "raw_videos"
        if not train:
            self.load_data()
        else:
            self.load_train()

    def filter_data(self, ignore_idx, data):
        ret = list()
        for idx, item in enumerate(data):
            if item['idx'] in ignore_idx:
                continue
            ret.append(item)
        return ret
    def load_data(self):
        val_video_path = self.video_path / "validation"
        # val file
        val_file = self.anno_path / "youcook2_val.csv"
        val_data = pd.read_csv(val_file)
        data = list()
        for i in range(len(val_data)):
            row = val_data.iloc[i]
            segment = row['segment'] # [46. 53.]
            query = row['sentence']
            recipe_type = row['recipe_type']
            video_path = val_video_path / str(recipe_type) / f"{row['youtube_id']}"
            # test whether video_path.mp4 or video_path.mkv exist
            mp4_path = video_path.with_suffix('.mp4')
            mkv_path = video_path.with_suffix('.mkv')
            if mp4_path.exists():
                video_path = mp4_path
            elif mkv_path.exists():
                video_path = mkv_path
            else:
                continue
            segment = segment.replace("[", "").replace("]", "").split()
            start_time, end_time = float(segment[0]), float(segment[1])
            duration = get_video_length(video_path)
            data.append({
                'idx': i,
                'data_path': str(video_path),
                'question': query,
                'answer': [start_time, end_time],
                'duration': duration,
            })
        # self.data = data
        ignore_idx = [1032, 1908, 3076]
        self.data = self.filter_data(ignore_idx, data)
 
        print(f"[{self.name}] length of data: {len(self.data)}")
        
    def load_train(self):
        val_video_path = self.video_path / "validation"
        # val file
        val_file = self.anno_path / "youcook2_val.csv"
        val_data = pd.read_csv(val_file)
        data = list()
        for i in range(len(val_data)):
            row = val_data.iloc[i]
            segment = row['segment'] # [46. 53.]
            query = row['sentence']
            recipe_type = row['recipe_type']
            video_path = val_video_path / str(recipe_type) / f"{row['youtube_id']}"
            # test whether video_path.mp4 or video_path.mkv exist
            mp4_path = video_path.with_suffix('.mp4')
            mkv_path = video_path.with_suffix('.mkv')
            if mp4_path.exists():
                video_path = mp4_path
            elif mkv_path.exists():
                video_path = mkv_path
            else:
                continue
            segment = segment.replace("[", "").replace("]", "").split()
            start_time, end_time = float(segment[0]), float(segment[1])
            duration = get_video_length(video_path)
            data.append({
                'idx': i,
                'data_path': str(video_path),
                'question': query,
                'answer': [start_time, end_time],
                'duration': duration,
            })
        # self.data = data
        ignore_idx = [1032, 1908, 3076]
        self.data = self.filter_data(ignore_idx, data)
 
        print(f"[Train] [{self.name}] length of data: {len(self.data)}")
        
        
        
    def load_data_old(self):
        self.video_path = self.db_path / "YouCookIIVideos"
        # val file
        val_file = self.anno_path / "youcook2_val.csv"
        val_data = pd.read_csv(val_file)
        data = list()
        for i in range(len(val_data)):
            row = val_data.iloc[i]
            segment = row['segment'] # [46. 53.]
            query = row['sentence']
            video_path = self.video_path / row['video_path']
            if not video_path.exists():
                continue
            segment = segment.replace("[", "").replace("]", "").split()
            start_time, end_time = float(segment[0]), float(segment[1])
            duration = get_video_length(video_path)
            data.append({
                'idx': i,
                'data_path': str(video_path),
                'question': query,
                'answer': [start_time, end_time],
                'duration': duration,
            })
        # self.data = data
        ignore_idx = [1032, 1908, 3076]
        self.data = self.filter_data(ignore_idx, data)
        # self.data = data[ignore_idx[-1]+1:]
        print(f"[{self.name}] length of data: {len(self.data)}")


class TextVQA(MMDS):
    def __init__(self, db_path = PROJECT_BASE / 'dataset/Images' / 'textvqa'):
        name = 'TextVQA'
        super().__init__(name)
        self.db_path = db_path
    
    def offload_data(self):
        # load the json file
        dataset = load_dataset("lmms-lab/textvqa", split="test")
        return dataset
    
    def offload_train(self):
        # load the json file
        dataset = load_dataset("lmms-lab/textvqa", split="train")
        self.offload(dataset, self.db_path / "train")
        return dataset
    
    def offload(self, ds, path: Path):
        images_dir = path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        ret = list()
        for idx, item in enumerate(tqdm(ds)):
            image_pil = item['image'] # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x730> object
            image_path = images_dir / f"{item['image_id']}.jpg"
            question = f"{item['question']}\n<image>"
            answer = item['answers'][0] if item['answers'] else ""
            image_pil.save(image_path)
            ds_item = sample_format_lambda(
                question=question,
                answer=answer,
                image_path=str(image_path),
                id=item['image_id'],
                path_prefix="images"  # relative path for the image
            )
            ret.append(ds_item)
        # save to json file
        output_file = path / "textvqa.json"
        with open(output_file, 'w') as f:
            json.dump(ret, f, indent=4)
        print(f"Processed TextVQA dataset, saved to {output_file}")
            
            

class DocVQA(MMDS):
    def __init__(self, db_path = PROJECT_BASE / 'dataset/Images' / 'docvqa'):
        name = 'DocVQA'
        super().__init__(name)
        self.db_path = db_path
    
    def reformat(self):
        # task 1: single page
        train_json = json.load(open(self.db_path / "task1/train_v1.0_withQT.json", 'r'))
        # train_json = json.load(open("train_v1.0_withQT.json", 'r'))
        train_data = train_json['data']
        ret = list()
        for idx, item in enumerate(tqdm(train_data)):
            # item = train_data[0]
            image_path = self.db_path / "task1" / item['image']
            question = f"{item['question']}\n<image>"
            answer = item['answers'][0] if item['answers'] else ""
            ds_item = sample_format_lambda(
                question=question,
                answer=answer,
                image_path=str(image_path),
                id=idx
            )
            ret.append(ds_item)
        # save to json file
        output_file = self.db_path / "task1" / "docvqa_task1.json"
        with open(output_file, 'w') as f:
            json.dump(ret, f, indent=4)
        print(f"Processed DocVQA task 1 dataset, saved to {output_file}")
        return ret
        
        

class Cambrian(MMDS):
    # path /home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Images/Cambrian-10M/Cambrian7M_withsystemprompt.jsonl
    def __init__(self, db_path = PROJECT_BASE / 'dataset/Images' / 'Cambrian-10M'):
        name = 'Cambrian'
        super().__init__(name)
        self.db_path = db_path
        
        
    def reformat(self):
        # load the jsonl file
        data = []
        with jsonlines.open(self.db_path / "Cambrian7M_withsystemprompt.jsonl") as reader:
            for item in reader:
                data.append(item)
        sources = set([item['source'] for item in data])
        # filter the data with sources in 'orca_994k.json
        d0 = [item for item in data if item['source'] in ['orca_994k.json']]
              
        ret = list()
        for idx, item in enumerate(tqdm(data)):
            image_path = self.db_path / "images" / item['image']
            question = f"{item['question']}\n<image>"
            answer = item['answer']
            ds_item = sample_format_lambda(
                question=question,
                answer=answer,
                image_path=str(image_path),
                id=idx
            )
            ret.append(ds_item)
        # save to json file
        output_file = self.db_path / "Cambrian7M_withsystemprompt_clean.json"
        with open(output_file, 'w') as f:
            json.dump(ret, f, indent=4)
        print(f"Processed Cambrian dataset, saved to {output_file}")
        return ret
    

class ActivityNetQA(MMDS):
    def __init__(self, db_path = PROJECT_BASE / 'dataset/Video/added' / 'ActivityNetQA', train=False):
        name = 'ActivityNetQA'
        super().__init__(name)
        self.db_path = db_path
        self.video_path = self.db_path / "v1-3/train_val"
        if not train:
            pass
        else:
            self.load_train()
        self.load_val()
        self.load_test()

    
    def load_val(self):
        # load the train_a.json and train_q.json
        train_answers = json.load(open(self.db_path / "val_a.json", 'r'))
        train_questions = json.load(open(self.db_path / "val_q.json", 'r'))
        self.data = list()
        for idx, item_qa in enumerate(tqdm(zip(train_questions, train_answers))):
            # print(item_qa)
            # exit(0)
            item, answer = item_qa
            assert item['question_id'] == answer['question_id'], f"Question ID mismatch: {item['question_id']} != {answer['question_id']}"
            video_path = self.video_path / f"v_{item['video_name']}.mp4"
            if not video_path.exists():
                continue
            question = f"<image>\n{item['question']}"
            answer = answer['answer']
            ds_item = {
                'idx': idx,
                'data_path': str(video_path),
                'question': question,
                'answer': answer,
            }
            self.data.append(ds_item)
        print(f"[Val] [{self.name}] length of data: {len(self.data)}")
    
    def load_test(self):
        # load the train_a.json and train_q.json
        train_answers = json.load(open(self.db_path / "test_a.json", 'r'))
        train_questions = json.load(open(self.db_path / "test_q.json", 'r'))
        self.data = list()
        for idx, item_qa in enumerate(tqdm(zip(train_questions, train_answers))):
            # print(item_qa)
            # exit(0)
            item, answer = item_qa
            assert item['question_id'] == answer['question_id'], f"Question ID mismatch: {item['question_id']} != {answer['question_id']}"
            video_path = self.video_path / f"v_{item['video_name']}.mp4"
            if not video_path.exists():
                continue
            question = f"<image>\n{item['question']}"
            answer = answer['answer']
            ds_item = {
                'idx': idx,
                'data_path': str(video_path),
                'question': question,
                'answer': answer,
            }
            self.data.append(ds_item)
        print(f"[Test] [{self.name}] length of data: {len(self.data)}")
    
    def load_train(self):
        # load the train_a.json and train_q.json
        train_answers = json.load(open(self.db_path / "train_a.json", 'r'))
        train_questions = json.load(open(self.db_path / "train_q.json", 'r'))
        self.data = list()
        for idx, item_qa in enumerate(tqdm(zip(train_questions, train_answers))):
            # print(item_qa)
            # exit(0)
            item, answer = item_qa
            assert item['question_id'] == answer['question_id'], f"Question ID mismatch: {item['question_id']} != {answer['question_id']}"
            video_path = self.video_path / f"v_{item['video_name']}.mp4"
            if not video_path.exists():
                continue
            question = f"<image>\n{item['question']}"
            answer = answer['answer']
            ds_item = {
                'idx': idx,
                'data_path': str(video_path),
                'question': question,
                'answer': answer,
            }
            self.data.append(ds_item)
        print(f"[Train] [{self.name}] length of data: {len(self.data)}")
        
    
    def reformat(self):
        train_answers = json.load(open(self.db_path / "train_a.json", 'r'))
        train_questions = json.load(open(self.db_path / "train_q.json", 'r'))
        ret = list()
        for idx, item_qa in enumerate(tqdm(zip(train_questions, train_answers))):
            item, answer = item_qa
            assert item['question_id'] == answer['question_id'], f"Question ID mismatch: {item['question_id']} != {answer['question_id']}"
            # item = train_questions[0]
            video_path = self.db_path / "v1-3/train_val" / f"v_{item['video_name']}.mp4"
            if not video_path.exists():
                continue
            question = f"<image>\n{item['question']}"
            answer = answer['answer']
            ds_item = sample_format_lambda(
                question=question,
                answer=answer,
                image_path=str(video_path),
                id=idx,
                path_prefix="v1-3/train_val"
            )
            ret.append(ds_item)
        # save to json file
        with open(self.db_path / "activitynetqa_train.json", 'w') as f:
            json.dump(ret, f, indent=4)
        print(f"Processed ActivityNetQA dataset, saved to {self.db_path / 'activitynetqa_train.json'}")
        



def test():
    import json
    path = "/home6/fzy/repos/EAGLE/dataset/ActivityNetCaps/val_1.json"
    val1 = json.load(open(path))
    print(list(val1.values())[0])

if __name__ == "__main__":
    # qvhl = QVHighlights(train=True)
    # print(qvhl[0])
    # textvqa = TextVQA()
    # textvqa.offload_train()
    
    docvqa = DocVQA()
    docvqa.reformat()
