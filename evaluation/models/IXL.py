import os
import time
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import requests

torch.set_grad_enabled(False)

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
class IXL_2d5():
    def __init__(self, args, patience=1, sleep_time=1):
        self.patience = patience
        self.sleep_time = sleep_time
        
        
        self.model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval().half()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer


        
    def get_response(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                assert os.path.exists(image_path)
                
                query = input_text + "\nLet's think step-by-step, perform reasoning first, then answer the question."
                image = [image_path]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    response, his = self.model.chat(self.tokenizer, query, image, do_sample=False, num_beams=1, use_meta=True)
            
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
