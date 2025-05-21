import numpy as np
import cv2

def extract_captcha_and_text(image: np.ndarray) -> tuple:
    """
    Cắt vùng CAPTCHA và văn bản hướng dẫn từ ảnh đầu vào.

    Args:
        image (np.ndarray): Ảnh đầu vào (BGR).

    Returns:
        tuple: (offset_x, offset_y, captcha_image, text_image).
               Trả về (None, None, None, None) nếu ảnh không hợp lệ.
    """
    if image is None:
        print("❌ Lỗi: Ảnh đầu vào không hợp lệ.")
        return None, None, None, None

    # Lấy kích thước ảnh
    height, width = image.shape[:2]
    print(f"🔍 Kích thước ảnh gốc: {width}x{height}")

    # Cắt vùng CAPTCHA (dựa trên tỷ lệ cố định)
    captcha_x = int(0.02 * width)
    captcha_y = int(0.215 * height)
    captcha_width = int(0.965 * width)
    captcha_height = int(0.67 * height)
    captcha_x_end = min(captcha_x + captcha_width, width)
    captcha_y_end = min(captcha_y + captcha_height, height)
    captcha_image = image[captcha_y:captcha_y_end, captcha_x:captcha_x_end]
    print(
        f"✅ Vùng CAPTCHA: Tọa độ ({captcha_x}, {captcha_y}), "
        f"Kích thước {captcha_image.shape[1]}x{captcha_image.shape[0]}"
    )

    # Cắt vùng văn bản hướng dẫn
    text_x = int(0.02 * width)
    text_y = int(0.015 * height)
    text_width = int(0.965 * width)
    text_height = int(0.20 * height)
    text_x_end = min(text_x + text_width, width)
    text_y_end = min(text_y + text_height, height)
    text_image = image[text_y:text_y_end, text_x:text_x_end]
    print(
        f"✅ Vùng văn bản: Tọa độ ({text_x}, {text_y}), "
        f"Kích thước {text_image.shape[1]}x{text_image.shape[0]}"
    )

    return captcha_x, captcha_y, captcha_image, text_image