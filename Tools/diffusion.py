import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

from huggingface_hub import notebook_login
notebook_login()

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

#prompt = "a photo of an astronaut riding a horse on mars"
#prompt = "a drive recorder footage of a monster"
#prompt = "a photograph of cat on a roof under fullmoon"
prompt = "a photograph of a cat having trouble on the toilet"
num_of_fig = 6

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = [prompt] * num_of_fig
images = []
with autocast("cuda"):
    images = pipe(prompt, guidance_scale=7.5)["sample"]

for i in range(num_of_fig):
	images[i].save(f"test%d.png"%(i))
