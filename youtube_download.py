import os
import youtube_dl
from multiprocessing import Pool, cpu_count
import shutil
import json

# 从json文件读取视频链接和时间段
with open('aud_cap_audiocaps.json', 'r') as file:
    video_data = json.load(file)

# 配置youtube-dl选项
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'audio/%(title)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}

def download_audio(data):
    video_id = data['video_id']
    base_video_id = video_id.rsplit('.', 1)[0]  # 去掉文件扩展名
    start_time, end_time = base_video_id.split('_')[1:3]
    start_time = float(start_time)
    end_time = float(end_time)
    output_filename = f"audio/{video_id}"
    
    ydl_opts['outtmpl'] = output_filename
    ydl_opts['postprocessor_args'] = [
        '-ss', str(start_time),
        '-to', str(end_time)
    ]
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={base_video_id.split('_')[0]}"])

# 创建进程池
with Pool(8) as p:
    p.map(download_audio, video_data)