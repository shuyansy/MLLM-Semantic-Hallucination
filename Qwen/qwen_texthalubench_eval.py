import os
import re
import math
import json
from PIL import Image
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load Qwen2.5-VL-3B model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "your/path/to/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("your/path/to/Qwen2.5-VL-3B-Instruct")

# Initialize evaluation counters
total_tp, total_fp, total_fn = 0, 0, 0

# Utility: Add newline before options like A., B., C., D.
def add_newline_before_options(text):
    return re.sub(r'(?<!\n)(?=[ABCD]\.)', '\n', text)

# Utility: Extract options (A-D) from question
def extract_options_from_question(question):
    pattern = r"([A-D])\.\s*(.+?)(?=(?:\n|$)[A-D]\.|$)"
    matches = re.findall(pattern, question, re.DOTALL)
    options, full_option_texts = {}, {}
    for letter, text in matches:
        clean_text = " ".join(text.strip().split())
        options[letter] = clean_text
        full_option_texts[letter] = f"{letter}.{clean_text}"
    return options, full_option_texts

# Utility: Predict which letters are selected in the model's response
def get_predicted_letters(response3, question):
    response3 = response3.replace("·", "").strip()
    predicted_words = re.split(r"[ \n]+", response3)

    options, full_option_texts = extract_options_from_question(question)
    matched_letters = []

    # Map options like "078" -> "A"
    num_to_letter = {}
    for letter, option_text in options.items():
        match = re.match(r"[A-Z]\.?(\w+)", option_text)
        if match:
            num = match.group(1)
            num_to_letter[num] = letter

    for word in predicted_words:
        word_clean = word.strip(".:;,-")
        for letter, option_text in options.items():
            full_text = full_option_texts[letter]
            if (
                word_clean == letter or
                word_clean.startswith(letter + ".") or
                word_clean == option_text or
                word_clean == full_text or
                (word_clean in num_to_letter and num_to_letter[word_clean] == letter)
            ):
                if letter not in matched_letters:
                    matched_letters.append(letter)
    return " ".join(matched_letters)




# ============ TEXT SPOTTING ============ #
spot_data = [json.loads(l) for l in open("Spotting.json", 'r', encoding='utf-8')]
for i in tqdm(range(1, 1430)):
    image_path = spot_data[i - 1]["filename"]
    question = "Read the all text in the image.Please reply in words."
    image = Image.open(image_path)
    orig_size = (image.width, image.height)

    # Resize (optional)
    # TOKEN_SIZE = 28
    # MAX_TOKENS = 8000
    # tokens_w, tokens_h = math.ceil(image.width / TOKEN_SIZE), math.ceil(image.height / TOKEN_SIZE)
    # num_tokens = tokens_w * tokens_h

    # if num_tokens > MAX_TOKENS:
    #     scale = math.sqrt((MAX_TOKENS * TOKEN_SIZE * TOKEN_SIZE) / (image.width * image.height))
    #     image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BICUBIC)
    #     image.save('/scqian/Qwen2.5-VL-3B-Instruct/resized.png')
    #     image_path = '/scqian/Qwen2.5-VL-3B-Instruct/resized.png'

    # Prepare prompt
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    # Locate image token positions
    image_token_id = 151655
    idx = (inputs.input_ids == image_token_id).nonzero(as_tuple=True)
    first_token_idx = idx[1][0].item()
    last_token_idx = idx[1][-1].item()
    
    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128,img_token_idx=first_token_idx, qs_token_idx=last_token_idx)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # Extract predictions
    real_texts = [t for t in spot_data[i - 1]["letter_group"] if t and t != "###"]
    index_colon = response.find(':')
    if index_colon != -1:
        response = response[index_colon + 1:].strip()
    match = re.search(r'\b(is|reads)\b (.+)', response)
    if match:
        response = match.group(2)
    predicted = list(set(re.split(r'[ \n]+', response.replace('"', '').replace('.', '').upper())))
    real_texts = list(set(t.upper() for t in real_texts))

    # Filter and deduplicate
    predicted = [p for p in predicted if p and 'TEXT' not in p]
    unmatched_real = real_texts.copy()

    for p in predicted:
        if p in unmatched_real:
            total_tp += 1
            unmatched_real.remove(p)
        else:
            total_fp += 1
    total_fn += len(unmatched_real)

# ============ TEXT UNDERSTANDING ============ #
under_data = [json.loads(l) for l in open("understanding.json", 'r', encoding='utf-8')]
under_questions = [json.loads(l) for l in open("understanding.json", 'r', encoding='utf-8')]

for i in tqdm(range(1, 301)):
    img_path = under_questions[i-1]["filename"]
    question_text = add_newline_before_options(under_questions[i-1]["letter_group"][0])

   
    image = Image.open(img_path)
    orig_size = (image.width, image.height)

    # Resize logic(optional)
    # TOKEN_SIZE = 28
    # MAX_TOKENS = 4000
    # tokens_w, tokens_h = math.ceil(image.width / TOKEN_SIZE), math.ceil(image.height / TOKEN_SIZE)
    # if tokens_w * tokens_h > MAX_TOKENS:
    #     scale = math.sqrt((MAX_TOKENS * TOKEN_SIZE * TOKEN_SIZE) / (image.width * image.height))
    #     image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BICUBIC)
    #     image.save('/scqian/Qwen2.5-VL-3B-Instruct/resized.png')
    #     img_path = '/scqian/Qwen2.5-VL-3B-Instruct/resized.png'
    #     orig_size = image.size

    
    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": question_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    # Locate image token positions
    image_token_id = 151655
    idx = (inputs.input_ids == image_token_id).nonzero(as_tuple=True)
    first_token_idx = idx[1][0].item()
    last_token_idx = idx[1][-1].item()

    # Generate prediction
    generated_ids = model.generate(
        **inputs, max_new_tokens=128,img_token_idx=first_token_idx, qs_token_idx=last_token_idx
    )
    output = processor.batch_decode(
        [g[len(inp):] for inp, g in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True
    )[0]
    predicted_letters = get_predicted_letters(output, question_text)
    real_labels = re.split(r'[ \n]+', under_questions[i - 1]["letter_group"][1])

    unmatched_real = real_labels.copy()
    for pred in predicted_letters:
        if pred in unmatched_real:
            total_tp += 1
            unmatched_real.remove(pred)
        else:
            total_fp += 1
    total_fn += len(unmatched_real)

# ============ FINAL METRICS ============ #
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives: {total_tp}")
print(f"False Positives: {total_fp}")
print(f"False Negatives: {total_fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save results
with open("qwen_combined_eval.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "precision": precision, "recall": recall, "f1_score": f1
    }, f, indent=4, ensure_ascii=False)
