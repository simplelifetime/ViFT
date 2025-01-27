import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import time


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
class llama_vision():
    def __init__(self, args, patience=1, sleep_time=1):
        self.patience = patience
        self.sleep_time = sleep_time
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        

    def get_response(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                assert os.path.exists(image_path)
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": input_text + "\nLet's think step-by-step, perform reasoning first, then answer the question."}
                    ]}
                ]
                image = Image.open(image_path)
                
                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # autoregressively complete prompt
                output = self.model.generate(**inputs, max_new_tokens=1024)
                response = self.processor.decode(output[0]).split('assistant<|end_header_id|>\n\n')[1].strip()
                
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
