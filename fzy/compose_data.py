import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import signal
import multiprocessing
import argparse
import logging
from typing import Union
from tqdm import tqdm
import time
from pathlib import Path
import json
from openai import OpenAI
from fzy.ds import ActivityNetCaps, Breakfast, Charades, QVHighlights, VALOR32K, YouCook2, sample_format_lambda
DATASET_BASE = Path("/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/added")


# 设置 base_url 来指向新的服务器地址
client = OpenAI(
    api_key="sk-s9J1EpRPlRg3KuUADb7a72F7D58740F997CaCdCbC30e3741",  # 替换为你的 API 密钥，如果从环境变量中获取可以省略
    base_url="http://10.162.159.8:53000/v1"  # 使用你自己的服务器地址
    # base_url="http://100.71.234.15:53000/v1"  # 使用你自己的服务器地址
)
# 第一个函数：生成文本请求
def generate_text(client, model, prompt, max_tokens=104, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    response = client.completions.create(
        model=model,  # 模型名称
        prompt=prompt,  # 输入的实际问题
        max_tokens=max_tokens,  # 限制返回的最大 tokens 数量
        temperature=temperature,  # 控制文本的生成多样性
        top_p=top_p,  # nucleus sampling
        frequency_penalty=frequency_penalty,  # 控制生成文本的重复度
        presence_penalty=presence_penalty  # 控制生成文本的内容广度
    )
    return response.choices[0].text

# 第二个函数：对话请求
def generate_chat_response(client, model, messages, max_tokens=1024, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    response = client.chat.completions.create(
        model=model,  # 替换为支持对话的模型
        messages=messages,  # 对话内容
        max_tokens=max_tokens,  # 限制返回的最大 tokens 数量
        temperature=temperature,  # 控制文本的生成多样性
        top_p=top_p,  # nucleus sampling
        frequency_penalty=frequency_penalty,  # 控制生成文本的重复度
        presence_penalty=presence_penalty  # 控制生成文本的内容广度
    )
    return response.choices[0].message.content

def generate_prompt_messages(caption_qa):
    """
    将 caption 风格的问答对转化为适合 prompting 的 messages 格式，用于生成 video QA。

    参数:
        caption_qa (dict): 包含 "question" 和 "answer" 字段的输入。

    返回:
        list[dict]: 可用于 openai/gpt 请求的 messages。
    """
    caption_question = caption_qa["question"]
    caption_answer = caption_qa["answer"]

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个多模态视频理解专家。请根据输入的 video caption，"
                "转换为一个清晰的视频问答（video QA）格式。"
                "输出应为 JSON 格式，包含 'question' 和 'answer' 两个字段，"
                "'question' 是针对视频内容的具体提问，'answer' 是基于 caption 原始语义推断的答案。"
            )
        },
        {
            "role": "user",
            "content": f"caption：{caption_question}"
        }
    ]

    return messages


def process_item(client, model, item):
    video_file = item['data_path']
    messages = generate_prompt_messages({
        "question": item['question'],
        "answer": item['answer']
    })

    try:
        chat_response = generate_chat_response(client, model, messages)
        chat_response = chat_response.strip().replace("```json", "").replace("```", "")
        response_json = json.loads(chat_response)
        question = response_json.get("question", "")
        answer = response_json.get("answer", "")
        return {
            'data_path': video_file,
            'question': f"During {item['answer'][0]}s to {item['answer'][1]}s, {question}",
            'answer': answer,
        }
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        return None


def convert_acnetcap_parallel(client, model="qwen2.5:72b", max_workers=8):
    ds = ActivityNetCaps(train=True)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, client, model, item) for item in ds]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                results.append(result)

    output_path = Path(DATASET_BASE / "processed/combined/acnetcap2qa.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Converted ActivityNetCaps to video QA format, saved to {output_path}")
    return results

def convert_acnetcap():
    ds = ActivityNetCaps(train=True)
    # 'data_path': str(video_file),
    # 'question': sentence,
    # 'answer': timestamps,
    # 'duration': v['duration'],
    ret = list()
    # model = "Qwen3-14B-AWQ"
    model = "qwen2.5:72b"
    for item in tqdm(ds):
        video_file = item['data_path']
        messages = generate_prompt_messages({
            "question": item['question'],
            "answer": item['answer']
        })
        # call api
        chat_response = generate_chat_response(client, model, messages)
        # parse response
        try:
            # remove ```json 
            chat_response = chat_response.strip().replace("```json", "").replace("```", "")
            # parse json
            response_json = json.loads(chat_response)
            question = response_json.get("question", "")
            answer = response_json.get("answer", "")
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {chat_response}")
            continue
        ret.append({
            'data_path': video_file,
            'question': f"During {item['answer'][0]}s to {item['answer'][1]}s, {question}",
            'answer': answer,
        })
    # save to DATASET_BASE/processed/combined/acnetcap2qa.json
    output_path = Path(DATASET_BASE / "processed/combined/acnetcap2qa.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(ret, f, indent=4)
    print(f"Converted ActivityNetCaps to video QA format, saved to {output_path}")
    return ret        


def sample_test():
    # 使用例子
    prompt = "如何提高一个多模态视频模型在某个评测集上的性能？"
    model = "Qwen3-14B-AWQ"
    # model = "qwen2.5:72b"

    # 调用第一个函数
    generated_text = generate_text(client, model, prompt)
    print(generated_text)

    # 对话消息示例
    # messages = [
    #     {"role": "user", "content": "我正在研究多模态视频模型的性能优化。"},
    #     {"role": "assistant", "content": "你可以尝试调整模型的超参数，或者增加训练数据。"},
    #     {"role": "user", "content": "目前，我的模型在某个评测集上的表现不够理想。如何进一步提高性能？"}
    # ]
    # 请你将输入的video caption问题转化为video qa问题的格式，将结果以JSON的形式返回，整理为"question"和"answer"两个字段，"question"为视频问题，"answer"为视频答案。
    
    
    # 调用第二个函数
    chat_response = generate_chat_response(client, model, messages)
    print(chat_response)


def combine_generated():
    acnetcap2qa_path = Path(DATASET_BASE / "processed/combined/acnetcap2qa.json")
    if not acnetcap2qa_path.exists():
        print(f"File {acnetcap2qa_path} does not exist.")
        return
    with open(acnetcap2qa_path, 'r') as f:
        acnetcap2qa_data = json.load(f)
    # 其他数据集的路径 dataset/Video/added/processed/combined/combined.json
    combined_path = Path(DATASET_BASE / "processed/combined/combined.json")
    if not combined_path.exists():
        print(f"File {combined_path} does not exist.")
        return
    with open(combined_path, 'r') as f:
        combined_data = json.load(f)
    # add acnetcap2qa_data to combined_data
    max_id = max(int(item['id']) for item in combined_data) if combined_data else 0
    for idx, item in enumerate(acnetcap2qa_data):
        item['id'] = idx + max_id + 1
        ds_item = sample_format_lambda(
            question=f"<image>\n{item['question']}",
            answer=item['answer'],
            image_path=item['data_path'],
            id=item['id']
        )
        combined_data.append(ds_item)
    # save to combined_path_out
    combined_path_out = Path(DATASET_BASE / "processed/combined/combined_acnetcap2qa.json")
    with open(combined_path_out, 'w') as f:
        json.dump(combined_data, f, indent=4)
    print(f"Combined data saved to {combined_path_out}")
    
    

if __name__ == "__main__":
    # convert_acnetcap()
    # convert_acnetcap_parallel(client, model="qwen2.5:72b", max_workers=32)
    combine_generated()
