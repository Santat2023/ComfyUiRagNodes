from . import load_images_utils
import chromadb
from . import s3_utils
import torch
import node_helpers
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import io

# --- подключение к Chroma и S3
client = chromadb.HttpClient(host="localhost", port=8000)

class DB_Load_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        try:
            collections = [c.name for c in client.list_collections()]
        except Exception as e:
            print(f"⚠️ Ошибка получения коллекций: {e}")
            collections = ["no_collections_found"]

        return {
            "required": {
                "user_initial_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Enter your prompt here"
                }),
                "collection_to_select": (collections,)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Ref image",)

    FUNCTION = "load_from_db"

    CATEGORY = "MyNodes"


    def load_from_db(self, user_initial_prompt, collection_to_select):
        file_bytes = load_images_utils.find_image_by_prompt(user_initial_prompt, collection_to_select)
        image = self.load_image(file_bytes)
        return (image, )
    
    def list_collections(self):
        return [c.name for c in client.list_collections()]
    
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
