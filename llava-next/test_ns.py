import os
import json
import re
import torch
import copy
import warnings
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
warnings.filterwarnings("ignore")

# ----------------------------- 模型初始化cd  -----------------------------

pretrained = "/your/path/to/LLaVA-NeXT/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"


tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,attn_implementation=None) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

# ----------------------------- Spotting -----------------------------

with open('Spotting.json', 'r', encoding='utf-8') as f:
    spot_data = [json.loads(line) for line in f]



spot_tp, spot_fp, spot_fn = 0, 0, 0

for i in range(1, 1430):
    
    filename = spot_data[i - 1]["filename"].lstrip("./")
    image_path = os.path.join("/scqian/NS-OCR", filename)#modify path
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llama_3"
    question = DEFAULT_IMAGE_TOKEN + "\nPlease tell me all of the words in the picture. You should only reply in those words."
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    image_sizes = [image.size]
    
    
    
    img_start_idx = (input_ids[0] == -200).nonzero(as_tuple=True)[0].item()
    question_start_idx = len(input_ids[0])-img_start_idx
    #breakpoint()
    output = model.generate(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes,do_sample=False, temperature=0, max_new_tokens=512,img_start_idx=img_start_idx,question_start_idx=question_start_idx)
    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print("#################################")
    
    real_groups = spot_data[i - 1]["letter_group"]
    real_groups = [text.upper() for text in real_groups if text and text != "###"]
    predicted_text = list(set(re.findall(r'\b\w+\b', response.upper())) - {"TEXT"})
    print(predicted_text)
    unmatched_real = real_groups.copy()
    for pred in predicted_text:
        if pred in unmatched_real:
            spot_tp += 1
            unmatched_real.remove(pred)
        else:
            spot_fp += 1
    spot_fn += len(unmatched_real)

spot_precision = spot_tp / (spot_tp + spot_fp) if (spot_tp + spot_fp) else 0
spot_recall = spot_tp / (spot_tp + spot_fn) if (spot_tp + spot_fn) else 0
spot_f1 = 2 * spot_precision * spot_recall / (spot_precision + spot_recall) if (spot_precision + spot_recall) else 0

with open("spot_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "TP": spot_tp, "FP": spot_fp, "FN": spot_fn,
        "Precision": round(spot_precision, 4),
        "Recall": round(spot_recall, 4),
        "F1": round(spot_f1, 4)
    }, f, indent=4, ensure_ascii=False)

print("🔎 SPOT识别完成！")

# ----------------------------- Understanding -----------------------------
def add_newline_before_options(text):
    return re.sub(r'(?<!\n)(?=[ABCD]\.)', '\n', text)

def extract_options_from_question(question: str):
    pattern = r"([A-D])\.\s*(.+?)(?=(?:\n|$)[A-D]\.|$)"
    matches = re.findall(pattern, question, re.DOTALL)
    options, full_options = {}, {}
    for letter, content in matches:
        clean = " ".join(content.strip().split())
        options[letter] = clean
        full_options[letter] = f"{letter}.{clean}"
    return options, full_options

def get_predicted_letters(response: str, question: str):
    response = response.strip()
    options, full_opts = extract_options_from_question(question)
    matched = []
    for letter, content in options.items():
        if content in response or full_opts[letter] in response or letter in response:
            matched.append(letter)
    return " ".join(matched)

with open("Understanding.json", "r", encoding="utf-8") as f:
    understand_data = [json.loads(line) for line in f]



under_tp, under_fp, under_fn = 0, 0, 0

for i in range(1, 301):
    try:
        filename = understand_data[i - 1]["filename"].lstrip("./")
        img_path = os.path.join("/scqian/NS-OCR", filename) #modify path
        question_raw = understand_data[i - 1]["letter_group"][0]
        question = add_newline_before_options(question_raw)

        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv = copy.deepcopy(conv_templates["llava_llama_3"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        img_start_idx = (input_ids[0] == -200).nonzero(as_tuple=True)[0].item()
        question_start_idx = len(input_ids[0])-img_start_idx
        #breakpoint()
        

        out = model.generate(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes,
                             do_sample=False, temperature=0, max_new_tokens=512,img_start_idx=img_start_idx,question_start_idx=question_start_idx)
        response = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        predicted = get_predicted_letters(response, question)
        real = re.split(r'[ \n]+', understand_data[i - 1]["letter_group"][1].strip())

        unmatched = real.copy()
        for p in predicted:
            if p in unmatched:
                under_tp += 1
                unmatched.remove(p)
            else:
                under_fp += 1
        under_fn += len(unmatched)

    except Exception as e:
        print(f"[UNDER ERROR @ {i}] {e}")

under_precision = under_tp / (under_tp + under_fp) if (under_tp + under_fp) else 0
under_recall = under_tp / (under_tp + under_fn) if (under_tp + under_fn) else 0
under_f1 = 2 * under_precision * under_recall / (under_precision + under_recall) if (under_precision + under_recall) else 0

with open("under_evaluation_results_ours.json", "w", encoding="utf-8") as f:
    json.dump({
        "TP": under_tp, "FP": under_fp, "FN": under_fn,
        "Precision": round(under_precision, 4),
        "Recall": round(under_recall, 4),
        "F1": round(under_f1, 4)
    }, f, indent=4, ensure_ascii=False)

print("📘 UNDER理解测试完成！")

# ----------------------------- output -----------------------------
average_f1 = (spot_f1 + under_f1) / 2
average_precision = (spot_precision + under_precision) / 2
average_recall = (spot_recall + under_recall) / 2

with open("overall_average_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "Average Precision": round(average_precision, 4),
        "Average Recall": round(average_recall, 4),
        "Average F1-Score": round(average_f1, 4)
    }, f, indent=4, ensure_ascii=False)

print("📊 平均指标计算完成！")
