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
from datasets import load_dataset
import torchvision.transforms as transforms
import cv2
from pathlib import Path
PROJECT_BASE = Path(__file__).resolve().parents[1]
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


sample_format_lambda = lambda question, answer, image_path, id, path_prefix="videos": {
    'id': str(id),
    'conversations': [
        {'from': 'human', 'value': question},
        {'from': 'gpt', 'value': answer}
    ],
    'image_abs': image_path, # /home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/added/ActivityNetCaps/v1-3/train_val/v_ehGHCYKzyZ8.mp4
    'image': f"{path_prefix}/{image_path.split('/')[-1]}" # v_ehGHCYKzyZ8.mp4
}


class VideoDS(Dataset):
    def __init__(self, name):
        self.name = name
        self.data = list()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ActivityNetCaps(VideoDS):
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
        

class ActivityNetQA(VideoDS):
    def __init__(self, db_path = PROJECT_BASE / 'dataset/Video/added' / 'ActivityNetQA', train=False):
        name = 'ActivityNetQA'
        super().__init__(name)
        self.db_path = db_path
        self.video_path = self.db_path / "v1-3/train_val"
        if not train:
            pass
        else:
            self.load_train()
    
    def load_train(self):
        # load the train_a.json and train_q.json
        train_answers = json.load(open(self.db_path / "train_a.json", 'r'))
        train_questions = json.load(open(self.db_path / "train_q.json", 'r'))
        self.data = list()
        for idx, item_qa in enumerate(tqdm(zip(train_questions, train_answers))):
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
        


class Breakfast(VideoDS):
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

            
        
    

class Charades(VideoDS):
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

                start_time = float(start_time)
                end_time = float(end_time)
                sample = {
                    "data_path": str(self.video_path / f"{video_id}.mp4"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": description,
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
                "answer": [item["start_time"], item["end_time"]]
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

                start_time = float(start_time)
                end_time = float(end_time)
                sample = {
                    "data_path": str(self.video_path / f"{video_id}.mp4"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": description,
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
                "answer": [item["start_time"], item["end_time"]]
            })
        self.data = ret
        print(f"[Train] [{self.name}] Loaded {len(data)} samples ")



class QVHighlights(VideoDS):
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
                'answer': line['relevant_windows'],
                'duration': line['duration'],
                'qid': line['qid'],
            })
        splits['train'] = deepcopy(data)
        
        print(f"[Train] [{self.name}] length of train data: {len(splits['train'])}")
        self.data = splits['train']


class VALOR32K(VideoDS):
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



class YouCook2(VideoDS):
    def __init__(self, db_path = DATASET_BASE / 'youcook2', train=False):
        name = 'youcook2'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path
        # self.video_path = self.db_path / "raw_videos"
        # self.video_path = Path("/home2/fzy/raw_videos")
        self.video_path = Path("/home2/fzy/raw_videos_8")
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
            # print(f"Loading video: {video_path}")
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
        print(f"Loading youcook2_val data")
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

class MVBench(VideoDS):
    def __init__(self, db_path = DATASET_BASE / 'MVBench', train=False):
        name = 'mvbench'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path / "test.json"
        self.video_path = self.db_path
        self.classes = ['action_sequence', 'moving_count', 'action_prediction', 'episodic_reasoning', 'action_antonym', 'action_count', 'scene_transition', 'object_shuffle', 'object_existence', 'fine_grained_pose', 'unexpected_action', 'moving_direction', 'state_change', 'object_interaction', 'character_order', 'action_localization', 'counterfactual_inference', 'fine_grained_action', 'moving_attribute', 'egocentric_navigation']
        self.load_data()
    
    def load_data(self):
        # load the json file
        self.data = list()
        self.test_json = json.load(open(self.anno_path, 'r'))
        for c, cprefix in self.test_json['root'].items():
            # c: class name
            # cpath: path to the json file
            if c not in self.classes:
                continue
            meta_c = self.test_json['meta'][c]
            for item in meta_c:
                video_path = self.video_path / cprefix / item['video']
                if not video_path.exists():
                    print(f"Video file not found: {video_path}")
                    continue
                question, answer = self.qa_template(item)
                ds_item = {
                    'data_path': str(video_path),
                    'question': question,
                    'answer': answer,
                    'class': c,
                }
                self.data.append(ds_item)

        print(f"[{self.name}] Total {len(self.data)} samples")
        # for c in self.classes:
        #     data = load_dataset(str(self.db_path), c)
        #     print(f"[{self.name}] Total {len(data)} samples")
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        question = question + "Only give the best option.\n"
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer


def test():
    import json
    path = "/home6/fzy/repos/EAGLE/dataset/ActivityNetCaps/val_1.json"
    val1 = json.load(open(path))
    print(list(val1.values())[0])

if __name__ == "__main__":
    m = MVBench()