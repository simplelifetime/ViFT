import os
import time
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from bunny.eval.llm_steer_bak import Steer


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
class Qwen2VL_LiTA():
    def __init__(self, args, patience=1, sleep_time=1):
        self.patience = patience
        self.sleep_time = sleep_time
        
        
        # default: Load the model on the available device(s)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )
        self.args = args
        
        min_pixels = 384 * 28 * 28
        max_pixels = 384 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(args.model_path,min_pixels=min_pixels, max_pixels=max_pixels)
        self.steered_model = Steer(self.model, self.processor.tokenizer)
        
    def get_response(self, image_path, input_text):
        patience = self.patience
        def process_query(query, image=None):
            if image:
                messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": query},
                    ],
                }
                ]   
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
                ]
                image_inputs, video_inputs = None, None
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            return inputs

        def generate_response(query, image):
            inputs = process_query(query, image)

            # Inference: Generation of the output
            generated_ids = self.steered_model.model.generate(**inputs, max_new_tokens=768)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return response
        
        
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                assert os.path.exists(image_path)
            
                def new_steering_method(tensor, coeff, try_keep_nr):
                    return coeff * tensor[0, -try_keep_nr: , :]

                self.steered_model.reset_all()
                inputs = process_query(input_text, None)
                try_keep_nr = inputs['input_ids'].size(1) - 13
                self.steered_model.add(layer_list=list(range(14, 28)), coeff=self.args.alpha, inputs=inputs, steering_method=new_steering_method, try_keep_nr=try_keep_nr)
                response = generate_response(input_text, image_path)

                response = response[0].strip()
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
