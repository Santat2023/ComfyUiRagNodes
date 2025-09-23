# ComfyUiRagCustomNodes

Этот пакет содержит кастомные ноды для [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

- Генерация изображений с помощью Stable Diffusion и LLM
- Загрузка и поиск изображений через ChromaDB и MinIO (S3)
- Интеграция с CLIP, Ollama, и другими инструментами

## Структура

- `custom_ksampler.py` — кастомный KSampler с поддержкой RAG
- `llm_node.py` — нода для работы с LLM (Ollama)
- `db_load_node.py` — загрузка изображений из базы
- `s3_utils.py` — утилиты для работы с MinIO/S3
- `load_images_utils.py` — утилиты для поиска и загрузки изображений

## Установка

Скопируйте папку в директорию `custom_nodes` вашего ComfyUI.

## Требования

- Python 3.10+
- ComfyUI
- MinIO/S3
- ChromaDB
- Ollama

## Использование

Добавьте ноды через интерфейс ComfyUI и настройте параметры согласно вашим задачам.