# SARK
## 📌 Overview

SARK is the first Style-Adaptive Reranker with Knowledge Prioritization, which prevents the model from overfitting to specific writing styles while retaining sensitivity to core evidence.

<div align=center>
<img src="https://github.com/lambdarw/SARK/blob/main/framework.png" width="70%" >
</div>

## 🧷 Data preparation
We evaluate our method on the stylistic NQ dataset. Examples are provided in the ```data/``` file.


## 🗒️ Model Files
This repository provides an example script for reranking inference using the SARK reranker model.

You can get the model files from the ```./model``` path, and download the ```model.safetensors``` file from quark site. (To comply with privacy policies, the links will be made public after acceptance.)

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
├── eval/                      # eval the reranker
│   ├── llm_eval.py            # use the llm eval the results
│   └── score_eval.py          # eval the model results score
├── requirement.txt
└── run.py
```

## 🚀 Quick Start

Step1: To set up, you can use the following command lines to set up Python 3.10 and PyTorch requirements:
``` python
pip install -r requirements.txt
```

Step2: If you want to run the program, follow this:

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
Please cite our repository if you use SARK in your work.
```bibtex
```
