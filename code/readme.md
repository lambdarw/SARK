## 📎Readme
This repository provides an example script for reranking inference using the SARK reranker model.

You can get the model files from the ```./model``` path, and download the ```model.safetensors``` file from [quark site](). (To comply with privacy policies, the links will be made public after acceptance.)

## 📂 Model Files
Here are the main files. 

Please run the model following the ```model/``` files and evaluate the results following ```eval/``` files.

```md
├── model/                      # all the model files
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
