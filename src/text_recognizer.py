import easyocr
import numpy as np

def extract_instruction_text(image: np.ndarray, use_gpu: bool = False) -> tuple:
    """
    Nhận dạng văn bản hướng dẫn từ ảnh vùng văn bản.

    Args:
        image (np.ndarray): Ảnh vùng văn bản (BGR).
        use_gpu (bool): Sử dụng GPU nếu True và có CUDA.

    Returns:
        tuple: (language, text) - Ngôn ngữ ("vi"/"en"/"unknown") và văn bản nhận dạng.
    """
    if image is None or not isinstance(image, np.ndarray):
        print("❌ Lỗi: Ảnh đầu vào không hợp lệ.")
        return "unknown", ""

    # Khởi tạo EasyOCR cho tiếng Việt và tiếng Anh
    reader = easyocr.Reader(['vi', 'en'], gpu=use_gpu)

    # Nhận dạng văn bản
    results = reader.readtext(image, detail=0)
    instruction_text = " ".join(results)

    # Xác định ngôn ngữ dựa trên ký tự
    detected_language = (
        "vi" if any(ord(c) > 127 for c in instruction_text)
        else "en" if instruction_text.strip()
        else "unknown"
    )

    return detected_language, instruction_text