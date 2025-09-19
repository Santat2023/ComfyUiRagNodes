import comfy.sample
import comfy.utils
import torch
import latent_preview
from comfy.comfy_types import IO
import requests
import folder_paths
import latent_preview
import node_helpers
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import io
from . import s3_example
from . import load_images
from .llm_node import LLM_Node
from .db_load_node import DB_Load_Node

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

class RAG_KSampler_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_initial_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Enter your prompt here"
                }),
                "vae": ("VAE", ),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "MyNodes"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, user_initial_prompt, vae, clip, model, seed, steps, cfg, sampler_name, scheduler, negative, latent_image, denoise=1.0):
        #file_bytes = s3_example.load_image_bytes_from_s3("5ae7e75f-abd5-41c8-8180-5282d776b2f7_ChatGPT Image 8. Sept. 2025, 21_32_04.png")
        file_bytes = load_images.find_image_by_prompt(user_initial_prompt)
        print("Loaded image bytes from S3, size:", len(file_bytes))
        image = self.load_image(file_bytes)
        print("Created image")
        result = self.encode(vae, image)[0]  # достаем dict
        print("Encoded image to latent, shape:", result["samples"].shape)
        
        modified_prompt_cond = self.call_llm(user_initial_prompt, clip)
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, modified_prompt_cond, negative, result, denoise=denoise)
    
    def encode(self, vae, pixels):
        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples":t}, )
    
    def load_image(self, file_bytes):
        #image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, io.BytesIO(file_bytes))
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        return output_image

    def call_llm(self, user_initial_prompt, clip):
        # Запрос к локальному Ollama
        try:
            print("start sending to LLM")

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:8b",  
                    "prompt": f"""
                    You are an assistant for Stable Diffusion. 
                    Rewrite the request into an expanded clean prompt. 
                    No explanations, no quotes, no markdown. 
                    Only a single line. Here is an example (!) of the format:

                    masterpiece, best quality, full body, small breast, 
                    2 young girls, jirai fashion with pink and black, looking at viewer, night, street, shop, store, neon

                    Now rewrite the following request in this format:
                    {user_initial_prompt}
                    """,
                    "stream": False
                },
                timeout=240  # ждём до 4 минут
            )
            # Логируем полный ответ сервера
            print("[LLM_Node] RAW RESPONSE:", response.text)

            if response.status_code == 200:
                data = response.json()
                if "response" in data and data["response"].strip():
                    result_text = data["response"].split("\n")[0].strip()
                else:
                    result_text = f"[LLM пустой ответ] {user_initial_prompt}"
            else:
                result_text = f"[Ошибка HTTP {response.status_code}] {user_initial_prompt}"

        except requests.exceptions.Timeout:
            result_text = f"[Ошибка Timeout] {user_initial_prompt}"
        except Exception as e:
            result_text = f"[Ошибка LLM: {e}] {user_initial_prompt}"

        print("[LLM_Node] FINAL PROMPT:", result_text)

        tokens = clip.tokenize(result_text)
        return clip.encode_from_tokens_scheduled(tokens)
    
NODE_CLASS_MAPPINGS = {
    "MyNodesForRAG": RAG_KSampler_Node,
    "MyNodesForLLM": LLM_Node,
    "MyNodesForDB": DB_Load_Node
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNodesForRAG": "RAG KSampler Node",
    "MyNodesForLLM": "LLM Node",
    "MyNodesForDB": "DB Load Node"
}
