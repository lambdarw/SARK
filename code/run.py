import json
import torch
import time
from transformers import AutoModel, AutoTokenizer
from typing import List
from tqdm import tqdm
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,5,6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


model_path = "./my_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model.to(device)
model.eval()

def rerank_batch_bulk(queries: List[str], all_passages: List[List[str]], batch_size=256):
    query_passage_pairs = []
    offsets = [0]
    for query, passages in zip(queries, all_passages):
        query_passage_pairs.extend([[query, passage] for passage in passages])
        offsets.append(offsets[-1] + len(passages))
    
    print("Warmup...")
    dummy_input = tokenizer(["test"]*2, ["test"]*2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
    with torch.no_grad():
        _ = model(**dummy_input)

    all_scores = []
    print(f"Start Benchmarking (Total pairs: {len(query_passage_pairs)})...")
    
    for i in tqdm(range(0, len(query_passage_pairs), batch_size), desc="Processing batches"):
        batch_pairs = query_passage_pairs[i : i + batch_size]
        batch_queries = [p[0] for p in batch_pairs]
        batch_docs = [p[1] for p in batch_pairs]
        current_bs = len(batch_queries)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        

        with torch.no_grad():
            inputs = tokenizer(
                batch_queries, 
                batch_docs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            if len(logits.shape) > 1 and logits.shape[1] > 1:
                scores = logits[:, 1].cpu().tolist()
            else:
                scores = logits.view(-1).cpu().tolist()
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        all_scores.extend(scores)
    
    return [all_scores[offsets[i]:offsets[i+1]] for i in range(len(queries))]


if __name__ == '__main__':
    input_file = './test_sampled_50.jsonl'
    output_file = './test_reranked_results.json'

    samples = []
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            samples.append({
                'question': sample['question'],
                'answers': sample['answers'],
                'top50': sample['rank_list']
            })
    
    queryList = [s['question'] for s in samples]
    passagesList = [s['top50'] for s in samples]


    scores_batch = rerank_batch_bulk(queryList, passagesList, batch_size=64)

    final_results = []
    for i in range(len(samples)):
        query_scores = scores_batch[i]
        query_passages = passagesList[i]
        
        scored_passages = list(zip(query_passages, query_scores))
        sorted_passages = sorted(scored_passages, key=lambda x: x[1], reverse=True)
        
        reranked_docs = [p[0] for p in sorted_passages]
        
        final_results.append({
            'question': samples[i]['question'],
            'answers': samples[i]['answers'],
            'top50': reranked_docs
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f'Finish and save in : {output_file}')