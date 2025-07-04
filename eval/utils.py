import os
# from dataclasses import dataclass

# @dataclass
# class VideoDataInput:
#     data_path: str | os.PathLike
#     question: str
#     answer: str
#     question_id: str = ''

DATASET_CACHE_TABLE = {
    'YouCookIIVideos':{
        'old': '.cache/huggingface/YouCookIIVideos/YouCookIIVideos',
        'new': 'dataset/eval/youcook2/videos'
    },
    'activitynetqa':{
        'old': '.cache/huggingface/activitynetqa/all_test',
        'new': 'dataset/eval/activitynetqa/videos'
    }
}

def replace_dir(input_path):
    replace_dataset = ''
    if "YouCookIIVideos" in input_path:
        replace_dataset = 'YouCookIIVideos'
    elif "activitynetqa" in input_path:
        replace_dataset = 'activitynetqa'
    
    output_path = ''
    if replace_dataset != '':
        old_str = DATASET_CACHE_TABLE[replace_dataset]['old']
        new_str = DATASET_CACHE_TABLE[replace_dataset]['new']
        output_path = input_path.replace(old_str, new_str)
        output_path = os.path.splitext(output_path)[0] + '.npy'

    return output_path
    

