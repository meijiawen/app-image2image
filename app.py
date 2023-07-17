import gradio as gr
from PIL import Image
from io import BytesIO
from openxlab.model import inference
# import modin.pandas as pd
# import torch
# import numpy as np
# from PIL import Image
# from diffusers import StableDiffusionInstructPix2PixPipeline

# model_id = "timbrooks/instruct-pix2pix"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#     model_id, torch_dtype=torch.float16, revision="fp16",
#     safety_checker=None) if torch.cuda.is_available(
#     ) else StableDiffusionInstructPix2PixPipeline.from_pretrained(
#         model_id, safety_checker=None)
# pipe = pipe.to(device)

# results = inference(model_repo='meijiawen1/inference-image2image',
#                     input=['./beignets-task-guide.png'])
# with open("./result.jpg", 'wb') as file:
#     file.write(results)

# def resize(value, img):
#     img = Image.open(img)
#     img = img.resize((value, value))
#     return img


def infer(source_img):
    # generator = torch.Generator(device).manual_seed(seed)
    # source_image = resize(512, source_img)
    # source_image.save('source.png')
    # image = pipe(
    #     instructions,
    #     image=source_image,
    #     guidance_scale=guide,
    #     image_guidance_scale=Strength,
    #     num_inference_steps=steps,
    #     generator=generator,
    # ).images[0]
    results = inference(model_repo='meijiawen1/inference-image2image',
                        input=[source_img])
    image = Image.open(BytesIO(results))
    return image


gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(source="upload",
                 type="filepath",
                 label="Raw Image. Must Be .png"),
        # gr.Textbox(
        #     label='Input Instructions. 77 Token (Keyword or Symbol) Maximum'),
        # gr.Slider(2, 15, value=7.5, label='Instructions Strength:'),
        # gr.Slider(
        #     1,
        #     20,
        #     value=5,
        #     step=1,
        #     label=
        #     "Number of Iterations: More take longer, but aren't always better"
        # ),
        # gr.Slider(label="Seed",
        #           minimum=0,
        #           maximum=987654321987654321,
        #           step=1,
        #           randomize=True),
        # gr.Slider(label='Original Image Strength:',
        #           minimum=1,
        #           maximum=2,
        #           step=.25,
        #           value=1.5)
    ],
    outputs='image',
    title="Image to Image",
    description="just test",
    article="模型中心推理服务test").queue(max_size=5).launch(max_threads=True,
                                                     debug=True)
