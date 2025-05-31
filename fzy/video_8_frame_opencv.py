import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames - 1
    frames = []

    for i in range(8):
        frame_id = int(i * total_frames / 8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps

def frames_to_video(frames, output_path, fps):
    if len(frames) == 0:
        return
    
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def process_video(args):
    video_path, output_path = args
    frames, fps = video_to_frames(video_path)
    frames_to_video(frames, output_path, fps)

def process_videos(input_folder, output_folder):
    tasks = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                video_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{video_name}.mp4")
                tasks.append((video_path, output_path))
    
    with Pool() as pool:
        list(tqdm(pool.imap(process_video, tasks), total=len(tasks)))

if __name__ == "__main__":
    input_folder = "/home/qinbosheng/HDD/HDD1/Code/MLLM/Video/train_video_and_instruction/train_600k_mp4"  # 替换为你的视频文件夹路径
    output_folder = "/home/qinbosheng/my_program/MLLM/EAGLE_LanguageBind/dataset/Video/train_600k_mp4"  # 替换为你的输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)
    process_videos(input_folder, output_folder)
