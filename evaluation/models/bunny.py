import os
import time
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from llm_steer_bak import Steer
import PIL


# verify response
def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        # print("Response Error")
        return False
    return True


# build bard class
class Bunny():
    def __init__(self, args, patience=1, sleep_time=1):
        self.patience = patience
        self.sleep_time = sleep_time
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        self.conv_mode = args.conv_mode
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type)
        self.steer = False
        self.args = args
        if args.steer:
            self.steer = True
            self.steered_model = Steer(self.model, self.tokenizer)
            print('using steer!')
            print(f'ALPHA: {self.args.alpha}')

        
    def process_query(self, query, image=None):
        if image:
            query = DEFAULT_IMAGE_TOKEN + '\n'  + query
            
            if not isinstance(image, PIL.PngImagePlugin.PngImageFile):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)[0].unsqueeze(0).cuda().to(dtype=self.model.dtype, device='cuda', non_blocking=True)

        else:
            image_tensor = None

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        return input_ids, image_tensor
    

    def get_response(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                if not isinstance(image_path, PIL.PngImagePlugin.PngImageFile):
                    assert os.path.exists(image_path)
                if self.steer:
                    self.steered_model.reset_all()
                    
                    input_ids, images = self.process_query(input_text, None)
                    try_keep_nr = input_ids.size(1)

                    def new_steering_method(tensor, coeff, try_keep_nr):
                        return coeff * tensor[0, -try_keep_nr: , :] 
                    
                    input_ids, images = self.process_query(input_text, None)
                    inputs = dict(input_ids=input_ids, images=images)
                    self.steered_model.add(layer_list=list(range(self.args.s_layer, self.args.e_layer)), coeff=self.args.alpha, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
                    
                    # input_ids, images = process_query(query, image)
                    # inputs = dict(input_ids=input_ids, images=images)
                    # steered_model.add(layer_list=list(range(14, 28)), coeff=-0.025, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
                    
                    input_ids, images = self.process_query(input_text, image_path)
                    inputs = dict(input_ids=input_ids, images=images)
                    self.steered_model.add(layer_list=list(range(self.args.s_layer, self.args.e_layer)), coeff=-self.args.beta, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
                    
                    input_ids, images = self.process_query(input_text, image_path)
                    with torch.inference_mode():
                        output_ids = self.steered_model.model.generate(
                            input_ids,
                            images=images,
                            do_sample=False,
                            max_new_tokens=4096,
                            use_cache=True,
                        )
                    input_token_len = input_ids.shape[1]
                    response = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    
                else:
                    input_ids, image_tensor = self.process_query(input_text, image_path)
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=False,
                            max_new_tokens=self.args.max_new_tokens,
                            use_cache=True,
                        )
                    input_token_len = input_ids.shape[1]
                    response = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    
                response = response.strip()
                if verify_response(response):
                    # print("# Verified Response")
                    # print(response)
                    return response
                else:
                    print(response)
            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""
