from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

model = '/nfs/huggingfacehub/Qwen/Qwen2.5-14B-Instruct'
# model = '/nfs/huggingfacehub/Qwen2.5-7B-Instruct'
# model = '/nfs/huggingfacehub/Llama-3.1-8B-Instruct'
# model = '/nfs/huggingfacehub/gemma-3-12b-it'

tokenizer = AutoTokenizer.from_pretrained(model)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

llm = LLM(model=model,  
        tensor_parallel_size=4,
        # max_model_len: informal 3000  6144  8192 
        # max_model_len: formal 3000(5/10) 6144(20) 7168(30) 9216(40) 11264(50)  
        max_model_len=11264,
        enforce_eager=True, 
        gpu_memory_utilization=0.9)


prompt = 'Transform the following formal paragraph in an informal tone.' \
             'Use everyday language, emotional expressions, or rhetorical flair. ' \
             'Ensure the output is 80-120 words. DO NOT INCLUDE ANYTHING ELSE. ' \
             'Now transform this: '


def load_concatenated_json(path):
    decoder = json.JSONDecoder()
    objs = []
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    idx = 0
    while idx < len(text):
        try:
            obj, idx = decoder.raw_decode(text, idx)
            objs.append(obj)
        except json.JSONDecodeError:
            idx += 1
    return objs

def write2file(outputs, wfile, res, type):
    i = 0
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if r"</think>" in generated_text:
            L = generated_text.split(r"</think>")
            if len(L) == 2:
                answer_content = L[1]
            else:
                answer_content = L[2]
        else:
            think_content = ""  
            answer_content = generated_text

        res[i][type] = answer_content.strip()
        i += 1
    
    with open(wfile, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    print("成功写入")

def myPrompt(question, context=None):
    if context: 
        prompt = "<Question>:"+question+ "\n<Context>:" + context +"\n<Answer>:"
        messages = [
            {"role": "system", "content": "You are an answer generation assistant. You need to answer the Question based on the Context with a brief response.  You should  only use a few words. \n[EXAMPLE] <Question>: What is the capital of France?\n <Context>: France is a beautiful country, its capital is Paris.\n<Answer>: Paris\n [/EXAMPLE]"},
            {"role": "user", "content": prompt}
        ]
    else: 
        prompt = "<Question>:"+question+"\n<Answer>:"
        messages = [
            {"role": "system", "content": "You are an answer generation assistant. You need to answer the question with a brief response using only a few words. \n [EXAMPLE]<Question>: What is the capital of France?\n <Answer>: Paris\n [/EXAMPLE]"},
            {"role": "user", "content": prompt},
        ]
    return messages

def myLLM(path, withContext=False, idx=1, multi=False, rerank=False):
    with open(path, 'r') as f:
        data = json.load(f)

    promptList = []
    res = []
    i = 0
    for sample in data:
        if withContext:
            if rerank :
                if multi:
                    text = sample['top50'][:idx]
                    contents = ""  
                    for content in text:
                        contents += content + "\n"
                else:
                    contents = sample['top50'][idx-1]
            else:
                if multi:
                    text = json.loads(sample['top50'][:idx])
                    contents = ""  
                    for content in text:
                        contents += content.get('contents', '') + '\n'
                else:
                    contents = sample['top50'][idx-1]

            messages = myPrompt(question=sample['question'], context=contents)
        else:
            messages = myPrompt(question=sample['question'])

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True
        )
        promptList.append(text)
        dic = {}
        dic["question"] = sample["question"]
        dic["answers"] = sample["answers"]
        if withContext:
            dic["context"] = contents
        res.append(dic)
        i += 1

    outputs = llm.generate(promptList, sampling_params)
    if multi:
        outPath = path[:-5] + '_multi'+ str(idx) + '_LLM.json'
    elif withContext:
         outPath = path[:-5] + '_top'+ str(idx) + '_LLM.json'
    else:
        outPath = path[:-5]  + '_LLM.json'
    
    write2file(outputs, outPath, res, "LLM")

def top1LLM(path, output_path, num=1, method='ours'):
    with open(path, 'r') as f:
        data = json.load(f)
    # data = load_concatenated_json(path)

    promptList = []
    res = []
    i = 0
    for sample in data:
        if method != 'DPR':
            contents = sample['top50']
        else:
            contents = sample['DPR']

        cnt = 0
        rag_contents = ''
        for content in contents:
            if cnt < num:
                rag_contents += content + '\n'
                cnt += 1
            else:
                break
        messages = myPrompt(question=sample['question'], context=rag_contents)


        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True
        )
        promptList.append(text)
        dic = {}
        dic["question"] = sample["question"]
        dic["answers"] = sample["answers"]
        dic["context"] = contents
        res.append(dic)
        i += 1

    outputs = llm.generate(promptList, sampling_params)
    
    write2file(outputs, output_path, res, "LLM")

def evalACC(path, withContext=False):
    with open(path, 'r') as f:
        data = json.load(f)

    promptList = []
    res = []
    i = 0
    def evalPrompt(question, answers, groundtruth, context=None):
        if context:
            pass
        else:
            prompt = "<Question>:" + question + "\n<LLM's Answer>:" + answers + "\n<Groundtruth>:" + groundtruth + "\n<Your Evaluation>:"
            messages = [
                {"role": "system", "content": "You are an answer evaluation assistant. Evaluate the <Question> by checking if the <LLM's Answer> matches the <Groundtruth>. Respond with 'Match' if identical, 'Partial' if partially correct, or 'Mismatch' if incorrect.\n Use only one word"},
                {"role": "user", "content": prompt}
            ]
        return messages
    
    i = 0
    for sample in data:
        groundtruth = ""
        if len(sample['answers']) > 1:
            for answer in sample['answers']:
                    groundtruth += answer
        else:
            groundtruth = sample['answers'][0]

        if withContext:
            messages = evalPrompt(question=sample['question'], answers=sample['LLM'], groundtruth=groundtruth, context=sample['question'])
        else:
            messages = evalPrompt(question=sample['question'], answers=sample['LLM'], groundtruth=groundtruth)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True
        )

        promptList.append(text)
        dic = {}
        dic["question"] = sample["question"]
        dic["answers"] = sample["answers"]
        dic['LLM'] = sample["LLM"]
        if 'context' in sample:
            dic['context'] = sample['context']
        res.append(dic)
        i += 1
    
    outputs = llm.generate(promptList, sampling_params)
    write2file(outputs, path[:-5]+"_eval.json", res, "LLM_EVAL")

def calculateAcc(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    acc = 0
    miss = 0
    part = 0
    for sample in data:
        if sample['LLM_EVAL'] == "Mismatch" or sample['LLM_EVAL'] == "Mismatch.":
            miss += 1
        elif sample['LLM_EVAL'] == "Match" or sample['LLM_EVAL'] == "Match." or sample['LLM_EVAL'] == "Match":
            acc += 1
        else:
            part += 1
    print(acc, miss, part, acc + miss + part)


if __name__ == "__main__":
    base_path = ''
    model = 'qwen'

    methods = ['ours']  
    ttype = 'informal'  # informal  formal  sample50
    nums = [5, 10, 20, 30, 40, 50]
    
    for method in methods:
        print(f"--------Evaling {method}--------")
        for num in tqdm(nums):
            input_path = f'{base_path}/{ttype}/{method}/test_rerank_style_{ttype}.json'
            output_path = f'{base_path}/{ttype}/{method}/{method}_test_{ttype}_top{num}_{model}.json'
            
            top1LLM(input_path, output_path, num)
            evalACC(output_path)   
    
