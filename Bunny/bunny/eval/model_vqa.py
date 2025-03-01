import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

from PIL import Image
import math
from bunny.eval.llm_steer_bak import Steer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type)
    steered_model = Steer(model, tokenizer)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    if args.steer:
        print('using steer!')
        
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        def process_query(query, image=None):
            if image:
                query = DEFAULT_IMAGE_TOKEN + '\n'  + query
                
                image = Image.open(image).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).cuda().to(dtype=model.dtype, device='cuda', non_blocking=True)

            else:
                image_tensor = None

            conv = conv_templates['qwen2'].copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            return input_ids, image_tensor
        
        empty_image_path = 'noise_images/empty_image.jpg'

        def generate_empty():
            import numpy as np
            random_noise = np.zeros([512, 512]) * 255
            noise_image = Image.fromarray(random_noise.astype('uint8')).convert('RGB')


            noise_image.save(empty_image_path)
            
        
        # cur_prompt = cur_prompt + '\nYou should provide a helpful, precise and concise response to the given query.'
            
            
        if args.steer:
            
            steered_model.reset_all()
            
            query = cur_prompt
            image = os.path.join(args.image_folder, image_file)
            
            input_ids, images = process_query(query, None)
            try_keep_nr = input_ids.size(1)

            def new_steering_method(tensor, coeff, try_keep_nr):
                return coeff * tensor[0, -try_keep_nr: , :] 
            
            # coeffs = [args.alpha]
            # import numpy as np
            # coeffs = np.array(coeffs)
            
            input_ids, images = process_query(query, None)
            inputs = dict(input_ids=input_ids, images=images)
            steered_model.add(layer_list=list(range(14, 28)), coeff=args.alpha, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
            
            # input_ids, images = process_query(query, image)
            # inputs = dict(input_ids=input_ids, images=images)
            # steered_model.add(layer_list=list(range(14, 28)), coeff=-0.025, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
            
            # generate_empty()
            # input_ids, images = process_query(query, empty_image_path)
            # inputs = dict(input_ids=input_ids, images=images)
            # steered_model.add(layer_list=list(range(7, 28)), coeff=-args.beta, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
            
            input_ids, images = process_query(query, image)

            with torch.inference_mode():
                output_ids = steered_model.model.generate(
                    input_ids,
                    images=images,
                    do_sample=False,
                    max_new_tokens=768,
                    use_cache=True,
                )
                

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            print(outputs)

        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=768,
                    use_cache=True)
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            print(outputs)


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--steer', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    args = parser.parse_args()

    eval_model(args)
