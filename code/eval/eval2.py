from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ExactMatch, RougeScore, BleuScore, StringPresence
from  bert_score import score, model2layers
import json
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

def bleu(pathList):
    for path in pathList:
        s = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                answer = item['answers'][0]
                sample = SingleTurnSample(
                    response = answer,
                    reference= item['LLM']
                )

                scorer = BleuScore()
                score = scorer.single_turn_score(sample)
                s += score

        print(s/len(data))


def sp(pathList):
    for path in pathList:
        s = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                answer = item['answers'][0]
                sample = SingleTurnSample(
                    response = answer,
                    reference= item['LLM']
                )

                scorer = StringPresence()
                score = scorer.single_turn_score(sample)
                s += score

        print(s/len(data))

def rouge(pathList):
    for path in pathList:
        s = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                answer = item['answers'][0]
                sample = SingleTurnSample(
                    response = answer,
                    reference= item['LLM']
                )

                scorer = RougeScore()
                score = scorer.single_turn_score(sample)
                s += score

        return "{:.4f}".format(s/len(data))

def em(pathList):
    for path in pathList:
        s = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                answer = item['answers'][0]
                sample = SingleTurnSample(
                    response = answer,
                    reference= item['LLM']
                )

                scorer = ExactMatch()
                score = scorer.single_turn_score(sample)
                s += score
        return "{:.4f}".format(s/len(data))

def calculateAcc(pathList):
    for path in pathList:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        acc = 0
        miss = 0
        part = 0
        for sample in data:
            if sample['LLM_EVAL'] == "Mismatch":
                miss += 1
            elif sample['LLM_EVAL'] == "Match":
                acc += 1
            elif sample['LLM_EVAL'] == "Partial":
                part += 1
        numSum = acc + miss + part

        return '{:.4f}'.format(acc / numSum)


def bert_score_calculation(pathList):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = 'bert-base-uncased'

    model2layers[model_path] = 12
    for path in pathList:
        total_f1 = 0.0
        num_samples = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cands = []
            refs = []
            batch_size = 8096

            for item in data:
                candidate = item['answers'][0]
                reference = item['LLM']
                cands.append(candidate)
                refs.append(reference)

                if len(cands) == batch_size:
                    P, R, F1 = score(cands=cands, refs=refs, lang='en', model_type='bert-base-uncased', device=device)
                    total_f1 += F1.sum().item()
                    num_samples += len(cands)
                    cands = []
                    refs = []

            if cands:
                P, R, F1 = score(cands=cands, refs=refs, lang='en', model_type='bert-base-uncased', device=device)
                total_f1 += F1.sum().item()
                num_samples += len(cands)

        avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0
        return "{:.4f}".format(avg_f1)


def eval_All(dataList, type='acc'):
    for pathList in dataList:
        if type == 'acc':
            print(pathList)
            calculateAcc(pathList)
        elif type == 'em':
            print(pathList)
            em(pathList)
        elif type == 'bs':
            print(pathList)
            bert_score_calculation(pathList)
        elif type == 'rouge':
            print(pathList)
            rouge(pathList)

if __name__ == "__main__":
    base_path = ''
    methods = ['ours']  
    ttype = 'informal'  # file type: informal  formal  sample50
    model = 'gemma4' # gemma4 gemma12 llama qwen3 qwen7 qwen14
    nums = [5, 10, 20, 30, 40, 50]

    for method in methods:
        print(f"Evaling {ttype} / Method {method} / Model {model}")
        results = {'acc':0, 'em':0, 'bertscore':0, 'rouge':0}
        for num in nums:
            print(f"-------------Top{num}-------------")
            path = [f'{base_path}/{ttype}/{method}/ours_test_{ttype}_top{num}_{model}_eval.json']
            
            results['acc'] = calculateAcc(path)
            results['em'] = em(path)
            results['bertscore'] = bert_score_calculation(path)
            results['rouge'] = rouge(path)
            print(results)
