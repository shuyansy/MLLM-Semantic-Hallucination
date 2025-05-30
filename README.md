# MLLM-Semantic-Hallucination


<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.17911-B21A1B)](https://arxiv.org/abs/2410.11779)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![GitHub Stars](https://img.shields.io/github/stars/shikiw/OPERA?style=social)](https://github.com/shikiw/OPERA/stargazers) -->

<p align="center">
  <a href="">📄arXiv</a> •
  <a href="">🤗HFPaper</a> •
</p>



This repository provides the official PyTorch implementation of the following paper: 
> [**When Semantics Mislead Vision:Mitigating Large Multimodal Models Hallucinationsin Scene Text Spotting and Understanding**]() <br>
> Yan Shu<sup>1</sup>, Hangui Lin<sup>2</sup>,   Yexin Liu<sup>3</sup>,
> Yan Zhang<sup>4,5</sup>,   Gangyan Zeng<sup>6</sup>,   Yan Li<sup>3</sup>,  Yu Zhou<sup>7</sup>,  Ser-Nam Lim<sup>8</sup>,   Harry Yang<sup>2</sup>,   Nicu Sebe<sup>1</sup>
> <br>
> <sup>1</sup>University of Trento (UNITN),   <sup>2</sup>University of International Relations (UIR), <sup>3</sup>The Hong Kong University of Science and Technology (HKUST),    <sup>4</sup>Institute of Information Engineering, Chinese Academy of Sciences (IIE, CAS),   <sup>5</sup>University of Chinese Academy of Sciences (UCAS),   <sup>6</sup>Nanjing University of Science and Technology (NJUST),  <sup>7</sup>Nankai University (NKU),  <sup>8</sup>University of Central Florida (UCF)


## Overview

<p align="center"><img src="img/method.png" alt="teaser" width="500px" /></p>

  Large Multimodal Models (LMMs) have achieved impressive progress in visualperception and reasoning. However, when confronted with visually ambiguous ornon-semantic scene text, they often struggle to accurately spot and understand thecontent, frequently generating semantically plausible yet visually incorrect answers,which we refer to as semantic hallucination. In this work, we investigate the un-derlying causes of semantic hallucination and identify a key finding: Transformerlayers in LLM with stronger attention focus on scene text regions are less prone to producing semantic hallucinations. Thus, we propose a training-free semantic hal-lucination mitigation framework comprising two key components: (1) ZoomText,a coarse-to-fine strategy that identifies potential text regions without external detec-tors; and (2) Grounded Layer Correction, which adaptively leverages the internalrepresentations from layers less prone to hallucination to guide decoding, correct-ing hallucinated outputs for non-semantic samples while preserving the semanticsof meaningful ones. To enable rigorous evaluation, we introduce TextHalu-Bench,a benchmark of over 1,730 samples spanning both semantic and non-semanticcases, with manually curated question–answer pairs designed to probe model hallu-cinations. Extensive experiments demonstrate that our method not only effectivelymitigates semantic hallucination but also achieves strong performance on publicbenchmarks for scene text spotting and understanding.


## Setup

### Qwen2.5-VL

We follow the official inplement of [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) and replace the transformer use following code.

```
cd MLLM-Semantic-Hallucination/Qwen
pip install .
```

### MiniMonkey

We follow the official inplement of [MiniMonkey](https://github.com/Yuliang-Liu/Monkey/tree/main/project/mini_monkey) and download their [official weight](https://huggingface.co/mx262/MiniMonkey).
Then we replace their code modeling_internlm2.py and modeling_minimonkey_chat.py with code [here](https://github.com/shuyansy/MLLM-Semantic-Hallucination/tree/master/minimonkey)

### LLaVA-NeXT 
We follow the steps below for inplement of LLaVA-NeXT .

```
cd llava-next
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install .
```



## Evaluation


### TextHalu-Bench

- Generate the MLLM's responses and save them in a jsonl file:
```bash
python chair_llava.py
```
<!-- Note: Please check out our released results in `log/chair_eval_results` for reproduction. -->

- Calculate CHAIR using the generated jsonl file:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```

### AMBER
- Generate the MLLM's responses and save them in a jsonl file:
```bash
python amber_llava.py
```

- Calculate metric score using the generated jsonl file:
```bash
python inference.py
```




### POPE
```bash
python pope_eval.py 
```
### MME
```bash
python mme_llava.py
```
###  Experiment's Results


## Reference Repositories
- DoLa: https://github.com/voidism/DoLa
- OPERA: https://github.com/shikiw/OPERA
- VCD: https://github.com/DAMO-NLP-SG/VCD
- LLaVA: https://github.com/haotian-liu/LLaVA
- MiniGPT4: https://github.com/Vision-CAIR/MiniGPT-4
