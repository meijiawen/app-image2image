import gradio as gr
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "timbrooks/instruct-pix2pix"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16") if torch.cuda.is_available() else StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def resize(value,img):
    img = Image.open(img)
    img = img.resize((value,value))
    return img

def infer(source_img, instructions, guide, steps, seed, Strength):
    generator = torch.Generator(device).manual_seed(seed)     
    source_image = resize(512, source_img)
    source_image.save('source.png')
    image = pipe(instructions, image=source_image,
            guidance_scale=guide, image_guidance_scale=Strength,
            num_inference_steps=steps, generator=generator,).images[0]
    return image

gr.Interface(fn=infer, inputs=[gr.Image(source="upload", type="filepath", label="Raw Image. Must Be .png"), 
    gr.Textbox(label = 'Prompt Input Text. 77 Token (Keyword or Symbol) Maximum'),
    gr.Slider(2, 15, value = 7, label = 'Guidance Scale'),
    gr.Slider(1, 20, value = 5, step = 1, label = 'Number of Iterations'),
    gr.Slider(label = "Seed", minimum = 0, maximum = 987654321987654321, step = 1, randomize = True), 
    gr.Slider(label='Strength', minimum = .1, maximum = 2, step = .05, value = .5)], 
    outputs='image', 
    description = "MUST Be .PNG and 512x512 or 768x768</b>) enter a Prompt, or let it just do its Thing, then click submit. 10 Iterations takes about ~900-1200 seconds currently. For more informationon about Stable Diffusion or Suggestions for prompts, keywords, artists or styles see https://github.com/Maks-s/sd-akashic", 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").queue(max_size=5).launch(debug=True)