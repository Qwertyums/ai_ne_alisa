import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Путь к локальной модели
MODEL_PATH = Path("./models/Qwen2.5-0.5B-Instruct")

# Глобальные переменные для модели и токенайзера
tokenizer = None
model = None

SYSTEM_PROMPT = """Ты — помощник кредитного аналитика банка для МСБ.
Тебе передано УЖЕ ПРИНЯТОЕ решение (rule_engine + ML).
Ты НЕ можешь изменить решение, пересчитать скор или придумать факты.
Пиши по-русски, эмпатично, без терминов SHAP/CatBoost/PD.
Структура ответа: 
1) Кратко, что решили (одобрено/отказ/на проверке). 
2) 2-3 ключевые причины простыми словами. 
3) Что бизнес может сделать, чтобы получить финансирование позже (советы).
Не обещай одобрение в будущем. Ответ должен быть не более 150 слов."""


def load_local_model():
    """Загружает модель, адаптируясь под доступное железо."""
    global tokenizer, model
    print("Загрузка локальной Qwen2.5-3B-Instruct...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    try:
        from transformers import BitsAndBytesConfig
        import torch
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Модель загружена в 4-битном режиме на GPU.")
        return
    except Exception as e:
        print(f"Не удалось загрузить на GPU ({e}). Переключаемся на CPU...")

    try:
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print("Модель успешно загружена на CPU (ожидание ответа займет ~5-10 сек).")
    except Exception as e:
        print(f"Критическая ошибка загрузки модели: {e}")
        print("Совет: если памяти совсем мало, скачайте модель Qwen2.5-0.5B-Instruct (весит ~1 ГБ)")


def generate_text(messages: list[dict], max_new_tokens: int = 300) -> str:
    """Генерирует текст с помощью локальной модели."""
    if model is None or tokenizer is None:
        raise RuntimeError("Модель не загружена. Вызовите load_local_model() при старте.")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()


def fallback_text(payload: dict) -> str:
    d = payload["decision"]
    lines = \
    {"APPROVE": "Предварительно одобряем.", "REVIEW": "Передано на ручную проверку.", "REJECT": "К сожалению, отказ."}[
        d["decision"]]
    return f"{lines}\nПричины: {', '.join(d['reasons'])}"


def explain(payload: dict) -> dict:
    """Главная функция, которая вызывается из роута /v1/explain"""
    if model is None:
        return {"mode": "template", "text": fallback_text(payload)}

    safe = {
        "decision": payload["decision"]["decision"],
        "reasons": payload["decision"]["reasons"],
        "conditions": payload["decision"]["conditions"],
        "top_drivers": payload["score"]["top_drivers"][:3],
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
    ]

    try:
        generated_text = generate_text(messages)
        return {"mode": "local_qwen", "text": generated_text}
    except Exception as e:
        return {"mode": "template_fallback", "text": fallback_text(payload), "error": str(e)}