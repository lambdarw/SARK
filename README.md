# SCOR
## 📌 Overview

SCOR is the first style-robust preference-aware reranking model, which integrates LLM-derived knowledge preference signals with style-invariant training, capturing content relevance under stylistic perturbations.

<div align=center>
<img src="https://github.com/lambdarw/SCOR/blob/main/framework.png" width="70%" >
</div>

## 🧷 Data preparation
We evaluate our method on the stylistic NQ dataset. Examples are provided in the ```data/``` file.


## 🗒️ Model Files
This repository provides an example script for reranking inference using the SCOR reranker model.

You can get the model files from the ```./model``` path, and download the ```model.safetensors``` file from [quark site](https://pan.quark.cn/s/d04abca325d9)

Please run the model following the ```model/``` files and evaluate the results following ```eval/``` files.

```md
├── model/                     # all the model files
│   ├── config.json
│   ├── model.safetensors
│   ├── modeling_joined.py
│   ├── sentencepiece.bpe.model
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── eval/                      # eval the reranker
    ├── llm_eval.py            # use the llm eval the results
    └── score_eval.py          # eval the model results score
```

## 🚀 Quick Start

Step1: To set up, you can use the following command lines to set up Python 3.10 and PyTorch requirements:
``` python
pip install -r requirements.txt
```

Step2: If you want to run the program, this is a batch size fun():

``` python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "YOUR_PATH"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
model.eval()

query = "what is the capital of China?"
passages = ["Beijing is the capital of China.", "Shanghai is the largest city in China.", "Chengdu is the largest city in China."]

queries_expanded = [query] * len(passages)

inputs = tokenizer(
    queries_expanded, 
    passages, 
    return_tensors="pt", 
    padding=True, 
    truncation=True, 
    max_length=512
)

inputs = {k: v.to('cuda') for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs)
    scores = logits.cpu().tolist()

print(scores)
```

## 📃 Citation
Please cite our repository if you use SCOR in your work.
```bibtex
```
