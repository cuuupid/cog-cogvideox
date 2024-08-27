import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
device = "cuda"

if not os.path.exists('./weights'):
  os.mkdir('./weights')

if not os.path.exists('./weights/glm-4'):
  os.mkdir('./weights/glm-4')
  tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
  ).to(device).eval()
  tokenizer.save_pretrained('./weights/glm-4')
  model.save_pretrained('./weights/glm-4')

if not os.path.exists('./weights/cog-video-x'):
  os.mkdir('./weights/cog-video-x')
  pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")
  pipe.save_pretrained("./weights/cog-video-x")
