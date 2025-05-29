import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import os
import re
from sklearn.metrics import precision_recall_fscore_support

total_tp, total_fp, total_fn = 0, 0, 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def dynamic_preprocess2(image_file,question,image, min_num=1, max_num=12, prior_aspect_ratio=None, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    new_target_ratios = []
    for i in target_ratios:
        if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
            new_target_ratios.append(i)
        else:
            continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        images=image_file
        queries=question
        #thumbnail_img = clip_api(images, queries, model_name="ViT-L-14-336")
        processed_images.append(thumbnail_img)
    return processed_images,target_aspect_ratio

def load_image(image_file, input_size=448, min_num=1, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio

def load_image2(image_file,question, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images,target_aspect_ratio = dynamic_preprocess2(image_file,question,image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num, prior_aspect_ratio=target_aspect_ratio)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values,target_aspect_ratio



# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
#path ='/scqian/Monkey/project/mini_monkey/MiniMonkey'#mx262/MiniMonkey'#'/scqian/Monkey/project/mini_monkey/work_dirs/minimonkey_chat/ft_ne_319' #
path = '/scqian/Monkey/project/mini_monkey/MiniMonkey'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)



# ======================= Spotting Evaluation =======================

spot_tp, spot_fp, spot_fn = 0, 0, 0

# Load Spotting data
file = open("Spotting.json", 'r', encoding='utf-8')
data = [json.loads(line) for line in file.readlines()]

for i in range(1, 1430):
    question = "Read the all text in the image.Please reply in words."
    new_path = filename = data[i - 1]["filename"]
    image_path = os.path.join(new_path)
    image = Image.open(image_path)

    pixel_values, target_aspect_ratio = load_image(image_path, min_num=4, max_num=12)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    pixel_values2, target_aspect_ratio = load_image2(
        image_path, question, min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio
    )
    pixel_values2 = pixel_values2.to(torch.bfloat16).cuda()

    pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

    generation_config = dict(do_sample=False, max_new_tokens=512)

    response3 = model.chat(tokenizer, pixel_values, target_aspect_ratio, question, generation_config, history=None, return_history=False)

    real_groups = data[i - 1]["letter_group"]
    real_groups = [item for item in real_groups if item != '' and item != '###']
    response3 = response3.replace("·", "")
    index = response3.find(':')
    if index != -1:
        response3 = response3[index + 1:].strip()

    match = re.search(r'\b(is|reads)\b (.+)', response3)
    if match:
        response3 = match.group(2)

    predicted_text = re.split(r'[ \n]+', response3.replace('"', '').replace('.', ''))
    predicted_text = [item for item in predicted_text if item != '']
    predicted_text = list(set(predicted_text))

    real_groups = [text.upper() for text in real_groups]
    predicted_text = [text.upper() for text in predicted_text]

    filter_keywords = ['text']
    seen = set()
    unique_list = [item for item in predicted_text if not any(k in item for k in filter_keywords) and not (item in seen or seen.add(item))]
    predicted_text = unique_list

    unmatched_real = real_groups.copy()
    for predicted in predicted_text:
        if predicted in unmatched_real:
            spot_tp += 1
            unmatched_real.remove(predicted)
        else:
            spot_fp += 1
    spot_fn += len(unmatched_real)

spot_precision = spot_tp / (spot_tp + spot_fp) if (spot_tp + spot_fp) > 0 else 0
spot_recall = spot_tp / (spot_tp + spot_fn) if (spot_tp + spot_fn) > 0 else 0
spot_f1 = 2 * spot_precision * spot_recall / (spot_precision + spot_recall) if (spot_precision + spot_recall) > 0 else 0

# ======================= Understanding Evaluation =======================

under_tp, under_fp, under_fn = 0, 0, 0

def add_newline_before_options(text):
    return re.sub(r'(?<!\n)(?=[ABCD]\.)', '\n', text)

def extract_answer(input_string):
    matches = re.findall(r'([A-D])\.', input_string)
    return ' '.join(matches)

# Load Understanding data
data2 = [json.loads(line) for line in open("/scqian/Monkey/project/mini_monkey/Understanding20.json", 'r', encoding='utf-8')]

for i in range(1, 301):
    print("*****************************************************************************************")
    img_path = data2[i - 1]['filename']
    question = data2[i - 1]['letter_group'][0]
    image_path = os.path.join(img_path)
    image = Image.open(image_path)

    pixel_values, target_aspect_ratio = load_image(image_path, min_num=4, max_num=12)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    pixel_values2, target_aspect_ratio = load_image2(
        image_path, question, min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio
    )
    pixel_values2 = pixel_values2.to(torch.bfloat16).cuda()

    question = add_newline_before_options(question)
    question += "\n\n\n\n\nAnswer the multiple-choice questions choose from four options (A B C D).Please output all possible options directly(EXAMPLE:A B C)without its content."

    pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

    generation_config = dict(do_sample=False, max_new_tokens=512)

    response3 = model.chat(tokenizer, pixel_values, target_aspect_ratio, question, generation_config, history=None, return_history=False)
    response3 = extract_answer(response3)

    real_groups = re.split(r'[ \n]+', data2[i - 1]["letter_group"][1])
    predicted_text = re.split(r'[ \n]+', response3)

    unmatched_real = real_groups.copy()
    for predicted in predicted_text:
        if predicted in unmatched_real:
            under_tp += 1
            unmatched_real.remove(predicted)
        else:
            under_fp += 1
    under_fn += len(unmatched_real)

under_precision = under_tp / (under_tp + under_fp) if (under_tp + under_fp) > 0 else 0
under_recall = under_tp / (under_tp + under_fn) if (under_tp + under_fn) > 0 else 0
under_f1 = 2 * under_precision * under_recall / (under_precision + under_recall) if (under_precision + under_recall) > 0 else 0

# ======================= Final Averaged F1 Score =======================

average_f1 = (spot_f1 + under_f1) / 2

# Save all results
results = {
    "Spotting Evaluation": {
        "TP": spot_tp,
        "FP": spot_fp,
        "FN": spot_fn,
        "Precision": round(spot_precision, 4),
        "Recall": round(spot_recall, 4),
        "F1": round(spot_f1, 4)
    },
    "Understanding Evaluation": {
        "TP": under_tp,
        "FP": under_fp,
        "FN": under_fn,
        "Precision": round(under_precision, 4),
        "Recall": round(under_recall, 4),
        "F1": round(under_f1, 4)
    },
    "Average F1 Score": round(average_f1, 4)
}

with open("/scqian/Monkey/project/mini_monkey/ablation_result/final_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Results saved to final_evaluation_results.json")
