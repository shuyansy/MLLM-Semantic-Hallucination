import json
import os
import re
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
# Initialize global counters for evaluation metrics
total_tp, total_fp, total_fn = 0, 0, 0

# Function to get token indices for bounding boxes
from tqdm import tqdm

import math
def get_patch_indices_from_bbox(
    bbox,                # (x_min, y_min, x_max, y_max) 归一化坐标
    original_size,       # 原始图像的尺寸 (height, width)
    patch_size=28        # 每个 token 对应 28x28 的区域
):
    x_min, y_min, x_max, y_max = bbox
    H_orig, W_orig = original_size

    # 计算图像的 grid 大小，假设每个 patch 的大小是 28x28
    grid_size_x = math.ceil(W_orig / patch_size)
    grid_size_y = math.ceil(H_orig / patch_size)

    # 计算 bounding box 对应的 pixel 坐标
    x_min = int(x_min * W_orig)
    x_max = int(x_max * W_orig)
    y_min = int(y_min * H_orig)
    y_max = int(y_max * H_orig)

    # 计算 bounding box 对应的 patch 索引
    patch_x_min = math.ceil(x_min / patch_size)  # 向上取整
    patch_x_max = math.floor((x_max - 1) / patch_size)  # 向下取整
    patch_y_min = math.ceil(y_min / patch_size)  # 向上取整
    patch_y_max = math.floor((y_max - 1) / patch_size)  # 向下取整

    # 生成 bounding box 所覆盖的所有 token 索引
    token_indices = [
        row * grid_size_x + col
        for row in range(patch_y_min, patch_y_max + 1)
        for col in range(patch_x_min, patch_x_max + 1)
    ]

    return token_indices

# Load the Qwen2.5 model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/scqian/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/scqian/Qwen2.5-VL-3B-Instruct")

# Utility functions for processing questions and extracting answers
def add_newline_before_options(text):
    return re.sub(r'(?<!\n)(?=[ABCD]\.)', '\n', text)

input_file = "/scqian/Qwen2.5-VL-3B-Instruct/STVQA/STVQA_DETECTED.json" 
output_file = "/scqian/Qwen2.5-VL-3B-Instruct/ablation_result/qwen2.5_stvqa_4.18.json"

# 加载测试数据
with open(input_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

results = []

file = open("/scqian/Monkey/project/mini_monkey/dectect_spot.json", 'r', encoding='utf-8')
data = []
for line in file.readlines():
    dic = json.loads(line)
    data.append(dic)

for item in tqdm(test_data["data"]):
    img_path = "/scqian/Qwen2.5-VL-3B-Instruct/STVQA/"+item["file_path"]
    question = item["question"]+"Answer me in a single word or phrase."
    question_id = item["question_id"]
    image_path = os.path.join(img_path )
    bbox=item["bbox"]
    if bbox==None or bbox==[]:
        bbox=[0.1,0.1,0.9,0.9]
        print("缺失值")
    # Process image and bounding box
    MAX_TOKENS = 4000
    TOKEN_SIZE = 28

    #image_path = os.path.join(img_path)
    image = Image.open(image_path)
    orig_size = (image.width, image.height)
    bbox_tokens = get_patch_indices_from_bbox(bbox, orig_size)
    # 当前 token 数量
    tokens_w = math.ceil(image.width / TOKEN_SIZE)
    tokens_h = math.ceil(image.height / TOKEN_SIZE)
    num_tokens = tokens_w * tokens_h

    if num_tokens > MAX_TOKENS:
        # 最大总像素面积（因为每 token 是 28x28）
        max_area = MAX_TOKENS * TOKEN_SIZE * TOKEN_SIZE
        current_area = image.width * image.height

        # 缩放比例（开方是因为面积 = w * h）
        scale = math.sqrt(max_area / current_area)

        # 新尺寸（四舍五入为整数）
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        # 等比例缩放
        image = image.resize((new_width, new_height), Image.BICUBIC)
        
        print(f"🔄 Image resized from {orig_size} to {image.size} to fit within {MAX_TOKENS} tokens.")
        resized_image_path='/scqian/Qwen2.5-VL-3B-Instruct/resized.png'
        image.save(resized_image_path)
        orig_size=image.size
        image_path=os.path.join(resized_image_path)
    else:
        print(f"✅ Image size {orig_size} is within {MAX_TOKENS} tokens, no resizing needed.")
    
    # x1, y1, x2, y2 = bbox
    # x1 *= orig_size[0]
    # x2 *= orig_size[0]
    # y1 *= orig_size[1]
    # y2 *= orig_size[1]
    # bbox = [x1, y1, x2, y2]
   
    #print(bbox_tokens)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Prepare inputs for the model
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    #breakpoint()
    image_token_id = 151655
    idx = (inputs.input_ids == image_token_id).nonzero(as_tuple=True)
    first_token_idx = idx[1][0].item()  # 第一个出现的 token 索引
    last_token_idx = idx[1][-1].item()  # 最后一个出现的 token 索引

    generated_ids = model.generate(**inputs,max_new_tokens=128,bbox_tokens=bbox_tokens,img_token_idx =first_token_idx,qs_token_idx=last_token_idx )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("question_id",question_id)
    print("answer:"+output_text[0])
    results.append({
        "question_id": question_id,
        "answer": output_text[0]
    })

# 输出为 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")