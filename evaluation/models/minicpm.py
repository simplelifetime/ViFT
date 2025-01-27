import os
import time
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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
class MiniCPM_model():
    def __init__(self, model_path, patience=1, sleep_time=1):
        self.patience = patience
        self.sleep_time = sleep_time
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16).eval().cuda() # sdpa or flash_attention_2, no eager
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def get_response(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                assert os.path.exists(image_path)
                image = Image.open(image_path).convert('RGB') # (jpeg, png, webp) are supported.
                msgs = [{'role': 'user', 'content': [image, input_text + "\nLet's think step-by-step, perform reasoning first, then answer the question."]}]
                response = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    top_p=1.0,
                    temperature=1.0
                )
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
    
    
    def get_multi_response(self, image_path, input_text, r_num):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                assert os.path.exists(image_path)
                image = Image.open(image_path).convert('RGB') # (jpeg, png, webp) are supported.
                msgs = [{'role': 'user', 'content': [image, input_text]}]
                responses = []
                for i in range(r_num):
                    response = self.model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=self.tokenizer,
                        top_p=1.0,
                        temperature=1.0
                    )
                    response = response.strip()
                    if verify_response(response):
                        # print("# Verified Response")
                        # print(response)
                        responses.append(response)
                    else:
                        print(response)
                return responses
            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""