import os
import io
import time
import argparse
import math
from tqdm import tqdm

import json
import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'

import sys
sys.path.append('../')
sys.path.append('../../evaluation')

from models import minicpm, bunny, llava_next, llava_ov, IXL, Qwen2VL, llama_v, Qwen2VL_LiTA
from datasets import load_dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
        

prompt_mc = 'Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end.\nQuestion: '
prompt_open = 'Please first conduct reasoning, and then answer the question and provide the final value, e.g., 1, 2.5, 300, at the end.\nQuestion: '


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--model_path', type=str, default='test')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard', 'minicpm', 'bunny', 'llava-next', 'llava-ov', 'IXL', 'QwenVL', 'llama_v', 'Qwen2VL_LiTA'])
    # Bunny setting
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--model-type", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--steer', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--s_layer', type=int, default=14)
    parser.add_argument('--e_layer', type=int, default=28)
    # Trunk setting
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    # load data
    
    dataset = load_dataset("MathLLMs/MathVision")

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    
    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}
        
    output_file = open(output_file, "w")

    # load model
    if 'minicpm' in args.model:
        model = minicpm.MiniCPM_model(args.model_path)
        print('Loading MiniCPM success')
    
    elif 'bunny' in args.model:
        model = bunny.Bunny(args)
        print('Loading Bunny success')
        
    elif 'llava-next' in args.model:
        model = llava_next.Llava_next(args)
        print('loading llava success')
        
    elif 'llava-ov' in args.model:
        model = llava_ov.Llava_ov(args)
        print('loading llava success')
        
    elif 'IXL' in args.model:
        model = IXL.IXL_2d5(args)
        print('loading IXL success')
        
    elif 'QwenVL' in args.model:
        model = Qwen2VL.Qwen2VL(args)
        print('loading Qwen2VL success')
        
    elif 'llama_v' in args.model:
        model = llama_v.llama_vision(args)
        print('loading llama_v success')
        
    elif 'Qwen2VL_LiTA' in args.model:
        model = Qwen2VL_LiTA.Qwen2VL_LiTA(args)
        print('loading Qwen2VL_LiTA success')
    
    # build final test pid list
    # test_pids = list(data.keys())
    
    test_pids = get_chunk(range(len(dataset['testmini'])), args.num_chunks, args.chunk_idx)
    
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)  

    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        d = dataset["testmini"][pid]
        image = f"image_path"
        question = d['question']
        if '<image2>' not in question:
            question = question.replace('<image1>', "")
        options = d['options']
        letters = ['A', 'B', 'C', 'D', 'E', 'F']
        if len(options) == 0:
            query = prompt_open + question
        else:
            query = prompt_mc + question
            query += '\nChoices:\n'
            for i, opt in enumerate(options):
                query += f"{letters[i]}: {opt}\n"
    
        results = d.copy()
        del results['decoded_image']
        results['query'] = query
        
        try:
            response = model.get_response(image, query)
            results["model_answer"] = response
        
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[pid]['error'] = e
    
    
        output_file.write(json.dumps(results) + "\n")
        
        output_file.flush()
        
    output_file.close()
        
