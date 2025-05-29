import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------- Model Initialization ----------------------
def load_model(checkpoint):
    """
    Load the language-vision model from Hugging Face checkpoint.
    """
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.pad_token_id
    return model, tokenizer

# ---------------------- Inference Utilities ----------------------
def format_query(img_path, question, checkpoint_name):
    """
    Format the multimodal query for the model. Different model require different templates.
    """
    if question == "Generate the detailed caption in English:" and "Monkey-Chat" not in checkpoint_name:
        return f'<img>{img_path}</img> Generate the detailed caption in English: '
    else:
        return f'<img>{img_path}</img> {question} Answer: '

def run_inference(model, tokenizer, image_path, question, checkpoint_name):
    """
    Run model inference on a single image-question pair.Different model require different process.
    """
    prompt = format_query(image_path, question, checkpoint_name)
    input_ids = tokenizer(prompt, return_tensors='pt', padding='longest').to('cuda')

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1
        )
    decoded = tokenizer.decode(output[0][input_ids.input_ids.size(1):], skip_special_tokens=True)
    return decoded.strip()

# ---------------------- Spot OCR Evaluation ----------------------
def evaluate_spot_ocr(model, tokenizer, data_path, root_path, checkpoint_name):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    tp, fp, fn = 0, 0, 0
    for i in tqdm(range(1, len(dataset) + 1), desc="Evaluating Spot OCR"):
        try:
            rel_path = dataset[i - 1]["filename"].lstrip("./")
            full_path = os.path.join(root_path, rel_path)
            question = "Please list all the words in this image. Only output the words, without explanations."
            response = run_inference(model, tokenizer, full_path, question, checkpoint_name)

            gt_words = [x.upper() for x in dataset[i - 1]["letter_group"] if x and x != "###"]
            pred_words = re.findall(r'\b\w+\b', response.upper())
            pred_words = list(set(pred_words) - {"TEXT"})

            unmatched = gt_words.copy()
            for word in pred_words:
                if word in unmatched:
                    tp += 1
                    unmatched.remove(word)
                else:
                    fp += 1
            fn += len(unmatched)
        except Exception as e:
            print(f"[Spot OCR Error @ {i}]: {e}")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return round(f1, 4)

# ---------------------- Multiple-Choice QA Evaluation ----------------------
def add_linebreaks(question):
    return re.sub(r'(?<!\n)(?=[ABCD]\.)', '\n', question)

def parse_options(text):
    pattern = r"([A-D])\.\s*(.+?)(?=(?:\n[A-D]\.|$))"
    return {m[0]: " ".join(m[1].strip().split()) for m in re.findall(pattern, text, re.DOTALL)}

def extract_answers(response, options):
    predicted = []
    for letter, opt in options.items():
        if opt in response or f"{letter}. {opt}" in response or letter in response:
            predicted.append(letter)
    return predicted

def evaluate_understanding(model, tokenizer, data_path, root_path, checkpoint_name):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    tp, fp, fn = 0, 0, 0
    for i in tqdm(range(1, len(dataset) + 1), desc="Evaluating Understanding QA"):
        try:
            rel_path = dataset[i - 1]["filename"].lstrip("./")
            full_path = os.path.join(root_path, rel_path)

            raw_q = dataset[i - 1]["letter_group"][0]
            answer_gt = dataset[i - 1]["letter_group"][1].strip().split()

            formatted_q = add_linebreaks(raw_q)
            response = run_inference(model, tokenizer, full_path, formatted_q, checkpoint_name)

            options = parse_options(formatted_q)
            predicted_letters = extract_answers(response, options)

            unmatched = answer_gt.copy()
            for p in predicted_letters:
                if p in unmatched:
                    tp += 1
                    unmatched.remove(p)
                else:
                    fp += 1
            fn += len(unmatched)
        except Exception as e:
            print(f"[Understanding Error @ {i}]: {e}")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return round(f1, 4)

# ---------------------- Main Evaluation Entry ----------------------
if __name__ == "__main__":
    # === Modify these paths as needed ===
    checkpoint = "echo840/Monkey-Chat"  # Replace with your model checkpoint and inference process
    root_path = "./TextHalu-Bench"
    spot_file = os.path.join(root_path, "Spotting.json")
    understanding_file = os.path.join(root_path, "Understanding.json")

    model, tokenizer = load_model(checkpoint)

    f1_spot = evaluate_spot_ocr(model, tokenizer, spot_file, root_path, checkpoint)
    f1_under = evaluate_understanding(model, tokenizer, understanding_file, root_path, checkpoint)
    avg_f1 = round((f1_spot + f1_under) / 2, 4)

    results = {
        "Spot OCR F1": f1_spot,
        "Understanding F1": f1_under,
        "Average F1": avg_f1
    }

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Evaluation Complete!")
    print(json.dumps(results, indent=2))
