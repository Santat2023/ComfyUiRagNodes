import torch
import clip
import chromadb
from . import s3_utils


# === Конфиг Chroma ===
client = chromadb.HttpClient(host="localhost", port=8000)
"""
try:
    images_collection = client.get_collection("images")
except:
    images_collection = client.create_collection("images")
"""

# === Модели ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

#blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)



# Эмбеддинг текста через CLIP
def embed_text(text: str):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        embedding = clip_model.encode_text(tokens)
        return embedding.cpu().numpy().tolist()


# Загрузка файла в S3 
def upload_to_s3(file_path: str, key: str):
    s3_utils.upload_image(file_path, key)
    url = f"{s3_utils.MINIO_ENDPOINT}/{s3_utils.BUCKET_NAME}/{key}"
    return url

def find_image_by_prompt(query: str, collection_name: str = "images"):
    res = search_images(query, collection_name, top_k=1)

    if not res["ids"] or not res["ids"][0]:
        raise ValueError("⚠️ Ничего не найдено по запросу!")

    # Берем первый результат
    id = res["ids"][0][0]
    meta = res["metadatas"][0][0]
    filename = meta["filename"]

    # Формируем ключ в S3
    object_name = f"{id}_{filename}"
    print(f"🎯 Найдено: ID={id}, файл={filename}, ключ={object_name}")

    return s3_utils.load_image_bytes_from_s3(object_name)

# Поиск по описанию
def search_images(query: str, collection_name: str , top_k: int = 5):
    embedding = embed_text(query)
    collection = client.get_collection(collection_name)
    res = collection.query(
        query_embeddings=embedding,
        n_results=top_k
    )
    return res