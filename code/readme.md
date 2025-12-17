# Readme
This repository provides an example script for reranking inference using the SPAR reranker model.

You can get the model files from the ```./model``` path, and download the ```model.safetensors``` file from [quark site](https://pan.quark.cn/s/d04abca325d9)

## Model Files
```md
├── model/                      # all the model files
│   ├── config.json
│   ├── model.safetensors
│   ├── modeling_joined.py
│   ├── sentencepiece.bpe.model
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── data/                       # the testing data examples
    └── test_sampled_50.jsonl
```
