import os
import io
import time
import argparse
import math
from tqdm import tqdm

import sys
sys.path.append('../')
from utilities import *

from models import claude, gpt, bard, minicpm, bunny, llava_next, llava_ov, IXL, Qwen2VL, Qwen, llama_v, Qwen2VL_LiTA

from build_query import create_query_data


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout
    
    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e
    
    # Restore the original stdout
    sys.stdout = old_stdout
    
    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()
    
    # Return the captured output or error
    return captured_output, error
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard', 'minicpm', 'bunny', 'llava-next', 'llava-ov', 'IXL', 'QwenVL', 'Qwen', 'llama_v', 'Qwen2VL_LiTA'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)  
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json') 
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')   
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', 
                        choices = ['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--model_path', type=str, default='test')
    parser.add_argument('--mutli_sampling', action='store_true', help='debug mode')
    # Bunny setting
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--model-type", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--steer', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--s_layer', type=int, default=0)
    parser.add_argument('--e_layer', type=int, default=28)
    # Trunk setting
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    
    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")                    
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

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

    # load model
    print(f"\nLoading {args.model}...")
    if args.model == 'bard':
        if args.key == '':
            print("Loading key from environment variable")
            key = os.environ['_BARD_API_KEY']
        else:
            key = args.key
        model = bard.Bard_Model(key)
    
    elif "gpt" in args.model:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.getenv("OPENAI_API_KEY")
        else:
            key = args.key
        model = gpt.GPT_Model(args.model, key)
    
    elif "claude" in args.model:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            key = args.key
        model = claude.Claude_Model(args.model, key)
        
    elif 'minicpm' in args.model:
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
    
    # elif 'Qwen' in args.model:
    #     model = Qwen.Qwen2(args)
    #     print('loading Qwen2 success')
    
    elif 'llama_v' in args.model:
        model = llama_v.llama_vision(args)
        print('loading llama_v success')
        
    elif 'Qwen2VL_LiTA' in args.model:
        model = Qwen2VL_LiTA.Qwen2VL_LiTA(args)
        print('loading Qwen2VL_LiTA success')
            
    print(f"Model loaded.")
    
    # build final test pid list
    # test_pids = list(data.keys())
    
    test_pids = get_chunk(list(data.keys()), args.num_chunks, args.chunk_idx)
    
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(args.data_dir, image)

        if args.debug:
            print("--------------------------------------------------------------")
        print(f"\nGenerating response for {pid}...")
        try:
            if args.mutli_sampling:
                response = model.get_multi_response(image_path, query, 10)
            else:
                response = model.get_response(image_path, query)
            # print(f"Response: {response}")
            results[pid] = problem
            results[pid]['query'] = query
            if args.shot_type == 'solution':
                results[pid]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[pid]['response'] = response
                results[pid]['execution'] = output
                results[pid]['error'] = str(error)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[pid]['error'] = e
    
        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")
