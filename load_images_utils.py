import torch
import clip
from qdrant_client import QdrantClient, models
from . import s3_utils

# Конфиг Qdrant
client = QdrantClient(host="localhost", port=6333)

#Модели
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# Эмбеддинг текста через CLIP
def embed_text(text: str):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        embedding = clip_model.encode_text(tokens)
        return embedding.cpu().numpy()[0]  # Qdrant ожидает 1D-вектор


# Загрузка файла в S3 
def upload_to_s3(file_path: str, key: str):
    s3_utils.upload_image(file_path, key)
    url = f"{s3_utils.MINIO_ENDPOINT}/{s3_utils.BUCKET_NAME}/{key}"
    return url


def find_image_by_prompt(query: str, collection_name: str = "images"):
    res = search_images(query, collection_name, top_k=1)

    if not res:
        raise ValueError("⚠️ Ничего не найдено по запросу!")

    # Берем первый результат
    hit = res[0]
    point_id = hit.id
    payload = hit.payload
    filename = payload.get("filename", "unknown")

    # Формируем ключ в S3
    object_name = f"{point_id}_{filename}"
    print(f"🎯 Найдено: ID={point_id}, файл={filename}, ключ={object_name}")

    return s3_utils.load_image_bytes_from_s3(object_name)


# Поиск по описанию
def search_images(query: str, collection_name: str, top_k: int = 5):
    embedding = embed_text(query)

    results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k
    )
    return results  # список моделей ScoredPoint
