import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
sys.path.append('..')
import timeout_decorator
import signal
import multiprocessing
import argparse
import logging
import torchvision.transforms as T
from PIL import Image
import numpy as np
import imageio
import cv2
from eval.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from typing import Union
from tqdm import tqdm
from torch.utils.data import Dataset
import time
from pathlib import Path
import json
from accelerate import Accelerator
from accelerate.utils import gather_object
import pickle
import os
from transformers import AutoTokenizer
from safetensors.torch import safe_open
from datasets import load_dataset
from train_video1 import ModelArguments
import shutil
import hashlib
import subprocess
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: decord not found, please install with: pip install decord")
    VideoReader = None

eval_logger = logging.getLogger("eval_mvbench")
CURRENT_DIR = Path(__file__).resolve().parent
MVBENCH_DIR = CURRENT_DIR.parent / "dataset/MVBench"

try:
    from eagle.model import *
    from eagle.model.builder import load_pretrained_model
    from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from eagle.conversation import conv_templates, SeparatorStyle
    from eagle.datasets.video_dataset import smart_tokenizer_and_embedding_resize
except ImportError:
    eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")
    raise ImportError("Failed to import eagle modules")

handle_stuck = False

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava-294-168-old", 
        help="Pretrained path of model"
    )
    parser.add_argument(
        "--model_name", 
        default="eagle", 
        help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--dataset_name",
        default=str(MVBENCH_DIR),
        help="Hugging Face dataset name for MVBench"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Specific subset of MVBench to evaluate (e.g., 'action_sequence', 'action_prediction'). If None, evaluates all subsets."
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--conv_template",
        default="llama3",
        help=("conv mode"),
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    # parser.add_argument('--distributed', action='store_true', help='Distributed evaluation')
    parser.add_argument(
        "--video_dir",
        default="./dataset/MVBench/video",
        help="Directory containing MVBench videos"
    )
    parser.add_argument(
        "--use_processed",
        action="store_true",
        help="Use processed/offloaded dataset instead of original videos"
    )
    parser.add_argument(
        "--processed_dir",
        default="./dataset/MVBench/processed",
        help="Directory containing processed dataset"
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=8,
        help="Number of segments/frames to extract from videos"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Resolution for video processing"
    )
    parser.add_argument(
        "--offload_dataset",
        action="store_true",
        help="Offload/preprocess dataset before evaluation"
    )
    args = parser.parse_args()
    return args

def gen_prompt(data, args):
    """Generate prompt for MVBench evaluation"""
    question = data.get("question", "")
    options = data.get("options", [])
    
    # Format the question with options for multiple choice
    if options and len(options) > 0:
        option_text = ""
        for i, option in enumerate(options):
            option_text += f"{chr(65+i)}. {option}\n"
        
        question_with_options = f"{question}\n{option_text}Please select the correct answer from the options above."
    else:
        question_with_options = question
    
    if DEFAULT_IMAGE_TOKEN not in question_with_options:
        question_with_options = DEFAULT_IMAGE_TOKEN + '\n' + question_with_options
    
    # Use conversation template
    conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], question_with_options)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    return prompt_question
# classes = ['action_sequence', 'moving_count', 'action_prediction', 'episodic_reasoning', 'action_antonym', 'action_count', 'scene_transition', 'object_shuffle', 'object_existence', 'fine_grained_pose', 'unexpected_action', 'moving_direction', 'state_change', 'object_interaction', 'character_order', 'action_localization', 'counterfactual_inference', 'fine_grained_action', 'moving_attribute', 'egocentric_navigation']
your_data_path = str((MVBENCH_DIR / "video").resolve())
data_list = {
    "Action Sequence": ("action_sequence.json", f"{your_data_path}/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", f"{your_data_path}/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", f"{your_data_path}/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", f"{your_data_path}/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", f"{your_data_path}/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", f"{your_data_path}/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", f"{your_data_path}/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", f"{your_data_path}/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", f"{your_data_path}/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", f"{your_data_path}/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", f"{your_data_path}/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", f"{your_data_path}/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", f"{your_data_path}/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", f"{your_data_path}/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", f"{your_data_path}/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", f"{your_data_path}/nturgbd/", "video", False),
    "Character Order": ("character_order.json", f"{your_data_path}/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", f"{your_data_path}/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", f"{your_data_path}/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", f"{your_data_path}/clevrer/video_validation/", "video", False),
}

data_dir = str(MVBENCH_DIR / "json")

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224, use_processed=False, processed_dir=None):
        self.data_list = []
        self.use_processed = use_processed
        self.processed_dir = processed_dir
        
        if use_processed and processed_dir and os.path.exists(os.path.join(processed_dir, "processed_index.json")):
            # Load processed dataset
            self.load_processed_dataset(os.path.join(processed_dir, "processed_index.json"))
        else:
            # Load original dataset
            for k, v in data_list.items():
                with open(os.path.join(data_dir, v[0]), 'r') as f:
                    json_data = json.load(f)
                for data in json_data:
                    self.data_list.append({
                        'task_type': k,
                        'prefix': v[1],
                        'data_type': v[2],
                        'bound': v[3],
                        'data': data
                    })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
        
    @classmethod
    def from_processed(cls, data_dir, data_list, processed_dir, num_segments=8, resolution=224):
        """Create MVBench_dataset instance from processed data"""
        return cls(data_dir, data_list, num_segments, resolution, use_processed=True, processed_dir=processed_dir)
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

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
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }
    
    def offload_dataset(self, output_dir="./dataset/MVBench/processed", num_frames=8):
        """
        Cut the bounded videos and convert all types to mp4 videos, according to the bound info, 
        save the offload videos and build new index data_list
        
        Args:
            output_dir (str): Directory to save processed videos
            num_frames (int): Number of frames to extract per video segment (used for frame sequences)
        
        Returns:
            dict: New data_list with processed video paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data_list = []
        processed_index = {}
        
        print(f"Starting dataset offloading to {output_dir}")
        print(f"Processing {len(self.data_list)} items...")
        
        for idx, item in enumerate(tqdm(self.data_list, desc="Processing videos")):
            try:
                task_type = item['task_type']
                data_type = item['data_type']
                prefix = item['prefix']
                bound = item['bound']
                data = item['data']
                
                # Get original video path
                original_video_path = os.path.join(prefix, data['video'])
                
                # Create unique identifier for this video segment
                video_id = data['video']
                if bound and item['bound']:
                    start_time = data.get('start', 0)
                    end_time = data.get('end', 0)
                    segment_id = f"{start_time:.2f}_{end_time:.2f}"
                else:
                    segment_id = "full"
                
                # Create hash for unique naming
                content_hash = hashlib.md5(f"{task_type}_{video_id}_{segment_id}".encode()).hexdigest()[:8]
                
                # Create output subdirectory for this task type
                task_output_dir = output_dir / task_type.replace(' ', '_').lower()
                task_output_dir.mkdir(parents=True, exist_ok=True)
                
                if data_type == 'video':
                    # Process video file
                    processed_path = self._process_video_segment(
                        original_video_path, task_output_dir, content_hash, 
                        bound, data, num_frames
                    )
                elif data_type == 'gif':
                    # Process gif file
                    processed_path = self._process_gif_segment(
                        original_video_path, task_output_dir, content_hash,
                        bound, data, num_frames
                    )
                elif data_type == 'frame':
                    # Process frame directory
                    processed_path = self._process_frame_segment(
                        original_video_path, task_output_dir, content_hash,
                        bound, data, num_frames
                    )
                else:
                    print(f"Unknown data type: {data_type}, skipping...")
                    continue
                
                if processed_path:
                    # Create new data entry
                    new_item = {
                        'task_type': task_type,
                        'prefix': str(task_output_dir),
                        'data_type': 'video',  # All processed data will be videos
                        'bound': False,  # No bounds needed for processed data
                        'data': {
                            **data,
                            'video': os.path.basename(processed_path),
                            'original_video': original_video_path,
                            'original_bound': bound,
                            'processed': True
                        }
                    }
                    processed_data_list.append(new_item)
                    
                    # Update index
                    key = f"{task_type}_{content_hash}"
                    processed_index[key] = {
                        'original_path': original_video_path,
                        'processed_path': processed_path,
                        'task_type': task_type,
                        'bound': bound,
                        'data': data
                    }
                    
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue
        
        # Save processed index
        index_file = output_dir / "processed_index.json"
        with open(index_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            serializable_index = {}
            for k, v in processed_index.items():
                serializable_index[k] = {
                    **v,
                    'processed_path': str(v['processed_path'])
                }
            json.dump(serializable_index, f, indent=2)
        
        print(f"Dataset offloading completed!")
        print(f"Processed {len(processed_data_list)} items")
        print(f"Index saved to {index_file}")
        
        # Update current data_list
        self.data_list = processed_data_list
        return processed_index
    
    def _process_video_segment(self, video_path, output_dir, content_hash, bound, data, num_frames):
        """Process video segment and save as mp4 video"""
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return None
                
            if VideoReader is None:
                print("VideoReader not available, skipping video processing")
                return None
            
            # Create output video path
            output_video_path = output_dir / f"video_{content_hash}.mp4"
            
            # Calculate time bounds
            if bound and 'start' in data and 'end' in data:
                start_time = data['start']
                end_time = data['end']
                
                # Use ffmpeg to extract video segment
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite output files
                    '-i', str(video_path),
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c:v', 'libx264',  # Use H.264 codec
                    '-c:a', 'aac',      # Use AAC audio codec
                    '-strict', 'experimental',
                    str(output_video_path)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg failed for {video_path}: {e}")
                    return None
            else:
                # No bounds, copy entire video and convert to mp4
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    str(output_video_path)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg failed for {video_path}: {e}")
                    return None
            
            return output_video_path
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None
    
    def _process_gif_segment(self, gif_path, output_dir, content_hash, bound, data, num_frames):
        """Process gif segment and save as mp4 video"""
        try:
            if not os.path.exists(gif_path):
                print(f"GIF file not found: {gif_path}")
                return None
            
            # Create output video path
            output_video_path = output_dir / f"gif_{content_hash}.mp4"
            
            # Calculate time bounds
            if bound and 'start' in data and 'end' in data:
                start_time = data['start']
                end_time = data['end']
                
                # Use ffmpeg to extract gif segment and convert to mp4
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(gif_path),
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',  # Ensure compatibility
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
                    str(output_video_path)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg failed for {gif_path}: {e}")
                    return None
            else:
                # No bounds, convert entire gif to mp4
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(gif_path),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    str(output_video_path)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg failed for {gif_path}: {e}")
                    return None
            
            return output_video_path
            
        except Exception as e:
            print(f"Error processing gif {gif_path}: {e}")
            return None
    
    def _process_frame_segment(self, frame_dir, output_dir, content_hash, bound, data, num_frames):
        """Process frame directory and convert to mp4 video"""
        try:
            if not os.path.exists(frame_dir):
                print(f"Frame directory not found: {frame_dir}")
                return None
                
            max_frame = len(os.listdir(frame_dir))
            fps = 3  # Default fps for frame sequences
            
            # Calculate frame indices based on bounds
            if bound and 'start' in data and 'end' in data:
                start_time = data['start']
                end_time = data['end']
                bound_tuple = (start_time, end_time)
            else:
                bound_tuple = None
            
            frame_indices = self.get_index(bound_tuple, fps, max_frame, first_idx=1)
            
            # Create temporary directory for selected frames
            temp_frames_dir = output_dir / f"temp_frames_{content_hash}"
            temp_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy specific frames to temp directory
            copied_frames = []
            for i, frame_index in enumerate(frame_indices):
                source_frame = os.path.join(frame_dir, f"{frame_index:05d}.jpg")
                if os.path.exists(source_frame):
                    target_frame = temp_frames_dir / f"{i+1:05d}.jpg"
                    shutil.copy2(source_frame, target_frame)
                    copied_frames.append(target_frame)
            
            if not copied_frames:
                print(f"No frames found for {frame_dir}")
                shutil.rmtree(temp_frames_dir)
                return None
            
            # Create output video path
            output_video_path = output_dir / f"frames_{content_hash}.mp4"
            
            # Convert frames to mp4 using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(temp_frames_dir / '%05d.jpg'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                str(output_video_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                # Clean up temporary directory
                shutil.rmtree(temp_frames_dir)
                return output_video_path
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg failed for frame sequence {frame_dir}: {e}")
                # Clean up temporary directory
                shutil.rmtree(temp_frames_dir)
                return None
            
        except Exception as e:
            print(f"Error processing frames {frame_dir}: {e}")
            return None
    
    def load_processed_dataset(self, processed_index_file):
        """Load previously processed dataset from index file"""
        try:
            with open(processed_index_file, 'r') as f:
                processed_index = json.load(f)
            
            # Rebuild data_list from processed index
            new_data_list = []
            for key, info in processed_index.items():
                processed_path = Path(info['processed_path'])
                if processed_path.exists():
                    new_item = {
                        'task_type': info['task_type'],
                        'prefix': str(processed_path.parent),
                        'data_type': 'video',  # All processed data are videos
                        'bound': False,  # No bounds needed for processed data
                        'data': {
                            **info['data'],
                            'video': processed_path.name,
                            'original_video': info['original_path'],
                            'original_bound': info['bound'],
                            'processed': True
                        }
                    }
                    new_data_list.append(new_item)
            
            self.data_list = new_data_list
            print(f"Loaded {len(new_data_list)} processed items from {processed_index_file}")
            return True
            
        except Exception as e:
            print(f"Error loading processed dataset: {e}")
            return False

def load_mvbench_dataset(args):
    """Load MVBench dataset using MVBench_dataset class"""
    try:
        # Check if we should use processed data
        processed_dir = getattr(args, 'processed_dir', None)
        use_processed = getattr(args, 'use_processed', False)
        
        if use_processed and processed_dir and os.path.exists(os.path.join(processed_dir, "processed_index.json")):
            # Load from processed data
            eval_logger.info(f"Loading processed MVBench dataset from {processed_dir}")
            dataset = MVBench_dataset.from_processed(
                data_dir=data_dir,
                data_list=data_list,
                processed_dir=processed_dir,
                num_segments=getattr(args, 'num_segments', 8),
                resolution=getattr(args, 'resolution', 224)
            )
        else:
            # Load original dataset
            eval_logger.info("Loading original MVBench dataset")
            
            # Filter data_list based on subset if specified
            filtered_data_list = data_list
            if args.subset:
                # Map subset names to data_list keys
                subset_mapping = {
                    'action_sequence': 'Action Sequence',
                    'action_prediction': 'Action Prediction',
                    'action_antonym': 'Action Antonym',
                    'fine_grained_action': 'Fine-grained Action',
                    'unexpected_action': 'Unexpected Action',
                    'object_existence': 'Object Existence',
                    'object_interaction': 'Object Interaction',
                    'object_shuffle': 'Object Shuffle',
                    'moving_direction': 'Moving Direction',
                    'action_localization': 'Action Localization',
                    'scene_transition': 'Scene Transition',
                    'action_count': 'Action Count',
                    'moving_count': 'Moving Count',
                    'moving_attribute': 'Moving Attribute',
                    'state_change': 'State Change',
                    'fine_grained_pose': 'Fine-grained Pose',
                    'character_order': 'Character Order',
                    'egocentric_navigation': 'Egocentric Navigation',
                    'episodic_reasoning': 'Episodic Reasoning',
                    'counterfactual_inference': 'Counterfactual Inference'
                }
                
                if args.subset in subset_mapping:
                    subset_key = subset_mapping[args.subset]
                    if subset_key in data_list:
                        filtered_data_list = {subset_key: data_list[subset_key]}
                        eval_logger.info(f"Loading subset: {args.subset} ({subset_key})")
                    else:
                        eval_logger.error(f"Subset key '{subset_key}' not found in data_list")
                        raise ValueError(f"Invalid subset: {args.subset}")
                else:
                    eval_logger.error(f"Unknown subset: {args.subset}")
                    raise ValueError(f"Invalid subset: {args.subset}")
            
            dataset = MVBench_dataset(
                data_dir=data_dir,
                data_list=filtered_data_list,
                num_segments=getattr(args, 'num_segments', 8),
                resolution=getattr(args, 'resolution', 224)
            )
        
        eval_logger.info(f"Loaded MVBench dataset with {len(dataset)} samples")
        eval_logger.info(f"Dataset info:\n{dataset}")
        
        # Convert dataset to list format for compatibility with existing evaluation code
        mvbench_data = []
        for idx in range(len(dataset)):
            try:
                item = dataset.data_list[idx]
                
                # Construct video path
                video_path = os.path.join(item['prefix'], item['data']['video'])
                
                # Create compatible data format
                data_item = {
                    'data_path': video_path,
                    'question': item['data'].get('question', ''),
                    'options': item['data'].get('candidates', []),  # MVBench uses 'candidates' instead of 'options'
                    'answer': item['data'].get('answer', ''),
                    'task_type': item['task_type'],
                    'video': item['data']['video'],
                    'idx': f"{item['task_type']}_{idx}",
                    'subset': item['task_type'].lower().replace(' ', '_').replace('-', '_'),
                    'data_type': item['data_type'],
                    'bound': item['bound'],
                    'original_data': item['data']  # Keep original data for reference
                }
                
                # Add start/end times if they exist
                if 'start' in item['data']:
                    data_item['start'] = item['data']['start']
                if 'end' in item['data']:
                    data_item['end'] = item['data']['end']
                
                mvbench_data.append(data_item)
                
            except Exception as e:
                eval_logger.warning(f"Error processing item {idx}: {e}")
                continue
        
        eval_logger.info(f"Converted {len(mvbench_data)} items to compatible format")
        return mvbench_data
        
    except Exception as e:
        eval_logger.error(f"Error loading MVBench dataset: {e}")
        raise e


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out!")

def get_image_tensor(data, image_processor, model, args):
    try:
        image_tensor = process_images(
            images=data["data_path"],
            image_processor=image_processor,
            model_cfg=model.config,
            modality='video',
        )
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
    except Exception as e:
        eval_logger.error(f"Error {e} in processing images")
        return None

# @timeout_decorator.timeout(3)
def get_image_tensor_timeout(data, image_processor, model, args, queue):
    try:
        image_tensor = process_images(
            images=data["data_path"],
            image_processor=image_processor,
            model_cfg=model.config,
            modality='video',
        )
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
        queue.put(("success", image_tensor))
    except Exception as e:
        queue.put(("error", str(e)))
        
            
@torch.no_grad()
def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    # accelerator = Accelerator()
    modality = 'video'
    # base_path = "/home6/fzy/repos/EAGLE/model/LLM/Llama-3.2-3B-Instruct"
    base_path = "./model/LLM/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = EagleLlamaForCausalLM.from_pretrained(
        base_path,
        low_cpu_mem_usage=True,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="<pad>"),
        tokenizer=tokenizer,
        model=model,
    )
    
    model_args_path = os.path.join(args.model_path, "model_args.pkl")
    if os.path.exists(model_args_path):
        with open(model_args_path, 'rb') as f:
            model_args: ModelArguments = pickle.load(f)
            model_args.evaluation = True  # Set evaluation to True for the model

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp="",
        modality=modality
    )
    tensors = {}

    safetensor_paths = [
        os.path.join(args.model_path, "model-00001-of-00002.safetensors"),
        os.path.join(args.model_path, "model-00002-of-00002.safetensors"),
    ]
    
    for safetensor_path in safetensor_paths:
        with safe_open(safetensor_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                if k in tensors:
                    print(f"警告：键 {k} 在多个safetensor文件中出现，后面的值会覆盖前面的值")
                tensors[k] = f.get_tensor(k)
    
    model.load_state_dict(tensors, strict=False)
    model = model.to(torch.float16)
    model.cuda()

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor.video_processor

    model.eval()
    if 'video' in args.model_path.lower() or '3d' in args.model_path.lower():
        modality = 'video'
    elif 'audio' in args.model_path.lower():
        modality = 'audio'
    print(f"Modality: {modality}, model type: {type(model)}, pretrained model: {args.model_path}")
    
    # Handle dataset offloading if requested
    if args.offload_dataset:
        eval_logger.info("Offloading/preprocessing dataset...")
        temp_dataset = MVBench_dataset(
            data_dir=data_dir,
            data_list=data_list,
            num_segments=args.num_segments,
            resolution=args.resolution
        )
        processed_index = temp_dataset.offload_dataset(
            output_dir=args.processed_dir,
            num_frames=args.num_segments
        )
        eval_logger.info(f"Dataset offloading completed. Processed {len(processed_index)} items.")
        # Set flag to use processed data
        args.use_processed = True
    
    # Load MVBench dataset
    eval_logger.info("Loading MVBench dataset...")
    test_dataloader = load_mvbench_dataset(args)
    eval_logger.info(f"Loaded {len(test_dataloader)} samples from MVBench")
        
    gen_list = list()
    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
    # model.get_vision_tower().config.num_frames = 16
    for i, data in enumerate(test_dataloader):
        # Check if video file exists
        if not os.path.exists(data["data_path"]):
            eval_logger.warning(f"Video file not found: {data['data_path']}")
            continue

        if not handle_stuck:
            try:
                # print(f"Loading {data['idx']} data")
                image_tensor = process_images(
                    images=[data["data_path"]],
                    image_processor=image_processor,
                    model_cfg=model.config,
                    modality=modality,
                )
                image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
            except Exception as e:
                eval_logger.error(f"Error {e} in processing images")
                continue
        else:
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=get_image_tensor_timeout, args=(data, image_processor, model, args, queue))
            process.start()
            # 设置超时时间（秒）
            timeout = 5
            process.join(timeout)

            if process.is_alive():
                print("Timeout: The task took too long to complete.")
                process.terminate()  # 超时后强制终止进程
                continue
            else:
                # 获取子进程的返回值
                result, data = queue.get()
            if result == "success":
                image_tensor = data
            else:
                print("Error occurred:", data)
                continue
            if image_tensor is None:
                continue
        
        # Extract video_grid_thw if available
        if hasattr(image_tensor, "video_grid_thw"):
            video_grid_thw = image_tensor['video_grid_thw']
            image_tensor = image_tensor['pixel_values_videos']
        else:
            video_grid_thw = None
            print("No video_grid_thw found, using None for video_grid_thw")
            
        prompt_question = gen_prompt(data, args)
        # print(prompt_question)
        
        input_ids = tokenizer_image_token(
            prompt_question, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        )
        pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # print(input_ids)
        input_ids = pad_sequence(
            tokenizer=tokenizer,
            input_ids=[input_ids], 
            batch_first=True, 
            padding_value=pad_token_ids
        ).to(args.device)
        attention_masks = input_ids.ne(pad_token_ids).to(args.device)

        gen_kwargs = {}
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        try:
            pbar.update(1)
            cont = model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=args.use_cache,
                modality=modality,
                video_grid_thw=video_grid_thw
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            # print(text_outputs)
        except Exception as e:
            continue
        #     eval_logger.error(f"Error {e} in generating")
        #     cont = ""
        #     text_outputs = [""]
        gen_list.append({
            "task_type": data.get("task_type", "mvbench"),
            "subset": data.get("subset", "unknown"),
            **data,
            "prediction": text_outputs[0],
        })
        
        # Save results periodically
        if args.output_path:
            output_file = args.output_path
        else:
            output_dir = CURRENT_DIR.parent / "output"
            output_dir.mkdir(exist_ok=True)
            subset_name = args.subset if args.subset else "all"
            output_file = output_dir / f"mvbench_{subset_name}_output.json"
            
        with open(str(output_file), "w") as f:
            json.dump(gen_list, f, indent=4)
    pbar.close()
    
    # Calculate and log accuracy by subset
    subset_accuracies, overall_accuracy = calculate_accuracy_by_subset(gen_list)
    
    # Save detailed results with accuracy metrics
    results_summary = {
        "overall_accuracy": overall_accuracy,
        "subset_accuracies": subset_accuracies,
        "total_samples": len(gen_list),
        "results": gen_list
    }
    
    # Save summary results
    if args.output_path:
        summary_file = str(Path(args.output_path).with_suffix('')) + "_summary.json"
    else:
        output_dir = CURRENT_DIR.parent / "output"
        output_dir.mkdir(exist_ok=True)
        subset_name = args.subset if args.subset else "all"
        summary_file = output_dir / f"mvbench_{subset_name}_summary.json"
    
    with open(str(summary_file), "w") as f:
        json.dump(results_summary, f, indent=4)
    
    eval_logger.info(f"Evaluation completed. Results saved to {output_file}")
    eval_logger.info(f"Summary with accuracy metrics saved to {summary_file}")
    eval_logger.info(f"Total samples processed: {len(gen_list)}")
    eval_logger.info(f"Final overall accuracy: {overall_accuracy:.4f}")



def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids


def calculate_accuracy(results):
    """Calculate accuracy for MVBench results"""
    if not results:
        return 0.0
    
    correct = 0
    total = len(results)
    
    for result in results:
        prediction = result.get("prediction", "").strip()
        ground_truth = result.get("answer", "").strip()
        
        # Extract the answer choice (A, B, C, D, etc.) from prediction
        prediction_choice = None
        for char in prediction.upper():
            if char in ['A', 'B', 'C', 'D', 'E']:
                prediction_choice = char
                break
        
        # Compare with ground truth
        if prediction_choice and prediction_choice == ground_truth.upper():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    eval_logger.info(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    
    return accuracy


def calculate_accuracy_by_subset(results):
    """Calculate accuracy for MVBench results grouped by subset"""
    if not results:
        return {}
    
    # Group results by subset
    subset_results = {}
    for result in results:
        subset = result.get("subset", "unknown")
        if subset not in subset_results:
            subset_results[subset] = []
        subset_results[subset].append(result)
    
    # Calculate accuracy for each subset
    subset_accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for subset, subset_data in subset_results.items():
        correct = 0
        total = len(subset_data)
        
        for result in subset_data:
            prediction = result.get("prediction", "").strip()
            ground_truth = result.get("answer", "").strip()
            
            # Extract the answer choice (A, B, C, D, etc.) from prediction
            prediction_choice = None
            for char in prediction.upper():
                if char in ['A', 'B', 'C', 'D', 'E']:
                    prediction_choice = char
                    break
            
            # Compare with ground truth
            if prediction_choice and prediction_choice == ground_truth.upper():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        subset_accuracies[subset] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        total_correct += correct
        total_samples += total
        
        eval_logger.info(f"Subset '{subset}': {correct}/{total} = {accuracy:.4f}")
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    eval_logger.info(f"Overall Accuracy: {total_correct}/{total_samples} = {overall_accuracy:.4f}")
    
    return subset_accuracies, overall_accuracy

def offload():
    """
    Offload/preprocess MVBench dataset by converting videos to mp4 format and processing all bounded segments.
    This function can be called independently to preprocess the dataset before evaluation.
    """
    # Parse command line arguments for offload operation
    parser = argparse.ArgumentParser(description="Offload MVBench dataset")
    parser.add_argument(
        "--processed_dir",
        default="./dataset/MVBench/processed",
        help="Directory to save processed dataset"
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=8,
        help="Number of segments/frames to extract from videos"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Resolution for video processing"
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Specific subset of MVBench to process (e.g., 'action_sequence', 'action_prediction'). If None, processes all subsets."
    )
    
    args = parser.parse_args()
    
    eval_logger.info("Starting MVBench dataset offloading...")
    eval_logger.info(f"Output directory: {args.processed_dir}")
    eval_logger.info(f"Number of segments: {args.num_segments}")
    eval_logger.info(f"Resolution: {args.resolution}")
    
    # Filter data_list based on subset if specified
    filtered_data_list = data_list
    if args.subset:
        # Map subset names to data_list keys
        subset_mapping = {
            'action_sequence': 'Action Sequence',
            'action_prediction': 'Action Prediction',
            'action_antonym': 'Action Antonym',
            'fine_grained_action': 'Fine-grained Action',
            'unexpected_action': 'Unexpected Action',
            'object_existence': 'Object Existence',
            'object_interaction': 'Object Interaction',
            'object_shuffle': 'Object Shuffle',
            'moving_direction': 'Moving Direction',
            'action_localization': 'Action Localization',
            'scene_transition': 'Scene Transition',
            'action_count': 'Action Count',
            'moving_count': 'Moving Count',
            'moving_attribute': 'Moving Attribute',
            'state_change': 'State Change',
            'fine_grained_pose': 'Fine-grained Pose',
            'character_order': 'Character Order',
            'egocentric_navigation': 'Egocentric Navigation',
            'episodic_reasoning': 'Episodic Reasoning',
            'counterfactual_inference': 'Counterfactual Inference'
        }
        
        if args.subset in subset_mapping:
            subset_key = subset_mapping[args.subset]
            if subset_key in data_list:
                filtered_data_list = {subset_key: data_list[subset_key]}
                eval_logger.info(f"Processing subset: {args.subset} ({subset_key})")
            else:
                eval_logger.error(f"Subset key '{subset_key}' not found in data_list")
                raise ValueError(f"Invalid subset: {args.subset}")
        else:
            eval_logger.error(f"Unknown subset: {args.subset}")
            raise ValueError(f"Invalid subset: {args.subset}")
    
    # Create MVBench dataset
    eval_logger.info("Creating MVBench dataset...")
    dataset = MVBench_dataset(
        data_dir=data_dir,
        data_list=filtered_data_list,
        num_segments=args.num_segments,
        resolution=args.resolution
    )
    
    eval_logger.info(f"Created dataset with {len(dataset)} items")
    eval_logger.info(f"Dataset info:\n{dataset}")
    
    # Perform offloading
    eval_logger.info("Starting dataset offloading/preprocessing...")
    processed_index = dataset.offload_dataset(
        output_dir=args.processed_dir,
        num_frames=args.num_segments
    )
    
    eval_logger.info(f"Dataset offloading completed!")
    eval_logger.info(f"Processed {len(processed_index)} video segments")
    eval_logger.info(f"Processed data saved to: {args.processed_dir}")
    eval_logger.info(f"Index file saved to: {args.processed_dir}/processed_index.json")
    
    # Verify processed dataset can be loaded
    try:
        eval_logger.info("Verifying processed dataset...")
        processed_dataset = MVBench_dataset.from_processed(
            data_dir, filtered_data_list, args.processed_dir, 
            num_segments=args.num_segments, resolution=args.resolution
        )
        eval_logger.info(f"Successfully verified processed dataset with {len(processed_dataset)} items")
    except Exception as e:
        eval_logger.error(f"Failed to verify processed dataset: {e}")
        raise e
    
    eval_logger.info("Offloading process completed successfully!")


if __name__ == "__main__":
    # args = parse_eval_args()
    # print(args)
    # evaluate(args=args)
    offload()
