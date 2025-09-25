import requests
from comfy.comfy_types import IO

class LLM_Node:
    def __init__(self):
        print("__init__ from LLM_NODE")
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_initial_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Enter your prompt here"
                }),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            },
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    RETURN_NAMES = ("cond",)

    FUNCTION = "call_llm"

    CATEGORY = "MyNodes"


    def call_llm(self, user_initial_prompt, clip):
        # Запрос к локальному Ollama
        try:
            print("start sending to LLM")

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2:7b",  
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
            # Логирование ответа сервера
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
        return (clip.encode_from_tokens_scheduled(tokens),)
