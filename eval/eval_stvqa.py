import os
import json
import torch
import warnings
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# -------------------- Configuration --------------------
CONFIG = {
    "model_name_or_path": "AIDC-AI/Ovis1.6-Gemma2-9B",   # Replace with any model
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_json": "./your/stvqa/annotation/file.json",
    "output_json": "./output.json",
    "image_base_path": "./images"
}

# -------------------- Model Loading --------------------
def load_model(config):
    print(f"🚀 Loading model: {config['model_name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.bfloat16 if config["device"] == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(config["device"])

    if hasattr(model, "get_text_tokenizer"):
        text_tokenizer = model.get_text_tokenizer()
    else:
        text_tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)

    visual_tokenizer = getattr(model, "get_visual_tokenizer", lambda: None)()

    return model, text_tokenizer, visual_tokenizer

# -------------------- Inference Function --------------------
def run_inference(model, text_tokenizer, visual_tokenizer, image_path, question):
    image = Image.open(image_path).convert("RGB")

    query = f"<image>\n{question} Answer me in a single word or phrase."

    # Preprocess inputs
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

    # Move data to device
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # Run generation
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )[0]

    # Decode and return
    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

# -------------------- Main Logic --------------------
def run_demo(config):
    model, text_tokenizer, visual_tokenizer = load_model(config)

    with open(config["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in tqdm(data["data"], desc="💡 Running Inference"):
        image_path = os.path.join(config["image_base_path"], item["file_path"])
        question = item["question"]
        question_id = item["question_id"]

        if not os.path.exists(image_path):
            print(f"[WARN] Image not found: {image_path}")
            results.append({"question_id": question_id, "answer": ""})
            continue

        answer = run_inference(model, text_tokenizer, visual_tokenizer, image_path, question)
        print(f"[{question_id}] {answer}")

        results.append({"question_id": question_id, "answer": answer})

    os.makedirs(os.path.dirname(config["output_json"]), exist_ok=True)
    with open(config["output_json"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Inference complete. Results saved to: {config['output_json']}")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    run_demo(CONFIG)
