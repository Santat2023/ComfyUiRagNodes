import boto3
from botocore.client import Config
from PIL import Image
import io

# Конфигурация MinIO
MINIO_ENDPOINT = "http://localhost:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "admin123"
BUCKET_NAME = "images"

# Создаём клиент
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

def upload_image(local_path: str, object_name: str):
    """Загрузка файла в MinIO"""
    s3.upload_file(local_path, BUCKET_NAME, object_name)
    print(f"✅ Файл {local_path} загружен как {object_name}")

def download_image(object_name: str, local_path: str):
    """Скачивание файла из MinIO"""
    s3.download_file(BUCKET_NAME, object_name, local_path)
    print(f"📥 Файл {object_name} скачан в {local_path}")

def load_image_from_s3(object_name: str):
    """Считывание изображения из S3 прямо в память"""
    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_name)
    file_bytes = response["Body"].read()   # байты файла
    image = Image.open(io.BytesIO(file_bytes))  # загрузка в PIL Image
    return image

def load_image_bytes_from_s3(object_name: str):
    """Считывание изображения из S3 прямо в память"""
    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_name)
    file_bytes = response["Body"].read()   # байты файла
    return file_bytes

def clear_bucket(bucket_name: str):
    """Удаляет все объекты из указанного бакета"""
    # получаем список объектов
    response = s3.list_objects_v2(Bucket=bucket_name)

    if "Contents" not in response:
        print("🪣 Бакет пуст")
        return

    # формируем список ключей для удаления
    objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

    # удаляем пачкой
    s3.delete_objects(Bucket=bucket_name, Delete={"Objects": objects_to_delete})
    print(f"🗑️ Удалено {len(objects_to_delete)} объектов из бакета '{bucket_name}'")

def create_bucket(bucket_name: str):
    """Создаёт бакет, если его нет"""
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"🪣 Бакет '{bucket_name}' уже существует")
    except:
        s3.create_bucket(Bucket=bucket_name)
        print(f"🪣 Бакет '{bucket_name}' создан")


"""if __name__ == "__main__":
    # Убедимся, что bucket существует
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
    except:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"🪣 Bucket '{BUCKET_NAME}' создан")

    # Пути
    #local_upload = "fox.jpg"
    local_upload = r"F:\MyProjects\Diploma\ChromaDb\fox.jpg"
    object_name = "animals/fox.jpg"
    local_download = "fox_downloaded.jpg"

    # Загружаем
    upload_image(local_upload, object_name)

    # Скачиваем обратно
    download_image(object_name, local_download)"""
