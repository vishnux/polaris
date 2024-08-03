import os
import uuid
import gradio as gr
import torch
from diffusers import DiffusionPipeline
import spaces

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
    variant="fp16"
)
pipe.to(device)

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
):
    seed = 1234
    generator = torch.Generator(device).manual_seed(seed)

    prompt = f"realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic, symmetric face and eyes, same eye color, full body pose"

    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": 1024,
        "height": 1024,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "num_images_per_prompt": 1,
    }
    
    images = pipe(**options).images
    image_path = save_image(images[0])
    return image_path

css = '''
.gradio-container {max-width: 750px !important; margin-left: auto; margin-right: auto;}
.generate-button {margin-top: 20px;}
.example-prompt-button {margin-top: 10px; margin-bottom: 10px; width: 100%; white-space: normal;}
'''

example_prompts = [
    "a beautiful woman wearing a pastel blue chiffon dress with floral embroidery, elegance and grace, smiling warmly",
    "a beautiful woman wearing a tailored black pantsuit with gold button accents, power and elegance, smiling warmly",
    "a beautiful woman wearing a vibrant red cocktail dress, sophistication and charm, smiling"
]

with gr.Blocks(theme="xiaobaiyuan/theme_brief", css=css) as demo:
    
    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Enter your prompt",
        lines=3
    )
    
    run_button = gr.Button("Generate", elem_classes="generate-button")
    
    result = gr.Image(label="Generated Image")
    
    gr.Markdown("### Example Prompts")
    with gr.Column():
        for example in example_prompts:
            btn = gr.Button(example, elem_classes="example-prompt-button")
            btn.click(lambda x=example: x, outputs=prompt)
    
    with gr.Accordion("Advanced Options", open=False):
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly, easy negative, bad hand, (low quality, worst quality:1.4), poorly drawn hands, bad anatomy, monochrome, { long body }, bad anatomy, liquid body, malformed, mutated, anatomical nonsense, bad proportions, uncoordinated body, unnatural body, disfigured, ugly, gross proportions, mutation, disfigured, deformed, { mutation}, {poorlydrawn}, bad hand, mutated hand, bad fingers, mutated fingers, badhandv4, liquid tongue, long neck, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, bad hairs, poorly drawn hairs, fused hairs, bad face, fused face, poorly drawn face, cloned face, big face, long face, bad eyes, fused eyes poorly drawn eyes, extra eyes, bad mouth, fused mouth, poorly drawn mouth, bad tongue, big mouth, bad perspective, bad objects placement",
            lines=3
        )
        guidance_scale = gr.Slider(
            label="Guidance Scale",
            minimum=1,
            maximum=20,
            step=0.1,
            value=7.5,
        )
        num_inference_steps = gr.Slider(
            label="Number of Inference Steps",
            minimum=10,
            maximum=100,
            step=1,
            value=30,
        )

    gr.Markdown("This app generates 4K hyper-realistic images based on your prompt.")

    run_button.click(
        fn=generate,
        inputs=[prompt, negative_prompt, guidance_scale, num_inference_steps],
        outputs=result,
    )

if __name__ == "__main__":
    demo.queue().launch()
