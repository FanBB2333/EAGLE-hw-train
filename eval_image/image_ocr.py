import os
from PIL import Image
import json
from tqdm import tqdm
from datasets import load_dataset
from paddleocr import PaddleOCR
import numpy as np


def main():
    # 设置输入图片根目录和输出json根目录
    input_dir = "./OCRBench_v2/EN_part"
    output_dir = "./OCRBench_v2/EN_part_ocr_json1"

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        lang="en"
    )

    def is_image_file(filename):
        ext = os.path.splitext(filename)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarray(i) for i in obj]
        return obj

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if is_image_file(file):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, input_dir)
                json_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".json")
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                try:
                    result = ocr.predict(img_path)
                    result = convert_ndarray(result[0].json['res'])
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=4)
                except:
                    print(f"图片处理失败: {img_path}")


def get_ocr_result(image: Image.Image, ocr) -> dict:
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarray(i) for i in obj]
        return obj
    
    try:
        # 将PIL图像转换为numpy数组
        img_array = np.array(image)
        # 进行OCR识别
        result = ocr.predict(img_array)
        # 转换numpy数组为普通Python对象
        result = convert_ndarray(result[0].json['res'])
        return result
    except Exception as e:
        print(f"OCR识别失败: {e}")
        return {}

def get_textvqa_results():
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        lang="en"
    )
    ocr_results = dict()
    for data in tqdm(ds, total=len(ds)):
        image = data['image'].convert('RGB')
        ocr_result = get_ocr_result(image, ocr)
        ocr_results[data['image_id']] = ocr_result

    with open("textvqa_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=4)
    return ocr_results

def get_chartqa_results():
    ds = load_dataset("lmms-lab/ChartQA", split="test")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        lang="en"
    )
    ocr_results = dict()
    for idx, data in tqdm(enumerate(ds), total=len(ds)):
        image = data['image'].convert('RGB')
        ocr_result = get_ocr_result(image, ocr)
        ocr_results[idx] = ocr_result

    with open("chartqa_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=4)
    return ocr_results

if __name__ == "__main__":
    # get_textvqa_results()
    get_chartqa_results()
    