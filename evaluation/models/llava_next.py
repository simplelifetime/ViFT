import os
import time
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests


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
class Llava_next():
    def __init__(self, args, patience=1, sleep_time=1):
        self.patience = patience
        self.sleep_time = sleep_time
        
        self.processor = LlavaNextProcessor.from_pretrained(args.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto") 
        

    def get_response(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                assert os.path.exists(image_path)
                image = Image.open(image_path)
                
                conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": input_text + "\nLet's think step-by-step, perform reasoning first, then answer the question."},
                    {"type": "image"},
                    ],
                },
                    ]
                
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
                # autoregressively complete prompt
                output = self.model.generate(**inputs, max_new_tokens=768)
                response = self.processor.decode(output[0], skip_special_tokens=True)
            
                response = response.strip()
                response = response.split('assistant\n\n\n')[1]
                
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
