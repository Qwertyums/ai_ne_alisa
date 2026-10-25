from huggingface_hub import snapshot_download
import os

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_DIR = "./models/Qwen2.5-0.5B-Instruct"

print(f"Скачивание {MODEL_ID} в {LOCAL_DIR}...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False, # Чтобы файлы лежали физически, а не ссылками
)

print("Модель успешно скачена!")