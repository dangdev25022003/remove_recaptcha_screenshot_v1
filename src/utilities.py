import json
import torch
import yaml

def load_config(config_path: str) -> dict:
    """
    Tải cấu hình từ file YAML.

    Args:
        config_path (str): Đường dẫn đến file YAML.

    Returns:
        dict: Nội dung cấu hình.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_translations(translations_path: str) -> dict:
    """
    Tải bản dịch từ file JSON.

    Args:
        translations_path (str): Đường dẫn đến file JSON.

    Returns:
        dict: Từ điển bản dịch.
    """
    with open(translations_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_translation(translations: dict, class_name: str, language: str) -> str:
    """
    Lấy bản dịch của tên lớp theo ngôn ngữ.

    Args:
        translations (dict): Từ điển bản dịch.
        class_name (str): Tên lớp cần dịch.
        language (str): Ngôn ngữ ("vi" hoặc "en").

    Returns:
        str: Tên đã dịch hoặc tên gốc nếu không có bản dịch.
    """
    try:
        return translations[class_name][language]
    except KeyError:
        print(f"⚠ Cảnh báo: Không tìm thấy bản dịch cho '{class_name}' trong '{language}'. Dùng tên gốc.")
        return class_name

def get_device(auto_detect: bool = True) -> str:
    """
    Xác định thiết bị xử lý (GPU hoặc CPU).

    Args:
        auto_detect (bool): Nếu True, ưu tiên GPU nếu có.

    Returns:
        str: "cuda" nếu dùng GPU, "cpu" nếu không.
    """
    if auto_detect and torch.cuda.is_available():
        print("✅ Sử dụng GPU (CUDA)")
        return 'cuda'
    print("✅ Sử dụng CPU")
    return 'cpu'