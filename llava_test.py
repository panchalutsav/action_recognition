import os

CACHE_DIR = "/mimer/NOBACKUP/groups/zijian/common/llava_model"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR + "/hub"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR + "/transformers"
os.environ["TORCH_HOME"] = CACHE_DIR


import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
else:
    print("Warning: HF_TOKEN environment variable not found.")


model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)



from huggingface_hub import constants
print("MODEL DOWNLOADED AT HF CACHE:", constants.HF_HUB_CACHE)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "list all the objects in the image"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
