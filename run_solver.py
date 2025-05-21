import cv2
import sys
import os
import numpy as np
from src.image_processor import detect_and_crop_captcha_region
from src.text_recognizer import extract_instruction_text
from src.captcha_handler import solve_3x3_captcha, solve_4x4_captcha
from src.utilities import load_config, load_translations, get_device
from src.crop_captcha_and_text import extract_captcha_and_text
from ultralytics import YOLO
import unicodedata

def load_screenshot(screenshot_path: str) -> np.ndarray:
    """
    Đọc ảnh chụp màn hình CAPTCHA và chuyển thành định dạng OpenCV (BGR).

    Args:
        screenshot_path (str): Đường dẫn đến file ảnh chụp màn hình.

    Returns:
        np.ndarray: Ảnh OpenCV (BGR) hoặc thoát chương trình nếu lỗi.
    """
    try:
        with open(screenshot_path, "rb") as file:
            screenshot_bytes = file.read()
        screenshot_array = np.frombuffer(screenshot_bytes, dtype=np.uint8)
        return cv2.imdecode(screenshot_array, cv2.IMREAD_COLOR)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy ảnh chụp màn hình tại '{screenshot_path}'")
        sys.exit(1)


def draw_and_display_results(
        result_image: np.ndarray,
        cell_centers: list,
        captcha_type: str
) -> None:
    """
    Vẽ các điểm đỏ tại tâm các ô được chọn và lưu ảnh kết quả.

    Args:
        result_image (np.ndarray): Ảnh gốc để vẽ kết quả.
        cell_centers (list): Danh sách tọa độ tâm các ô (x, y).
        captcha_type (str): Loại CAPTCHA ("3x3" hoặc "4x4").
    """
    for center_x, center_y in cell_centers:
        cv2.circle(
            result_image,
            (center_x, center_y),
            radius=5,
            color=(0, 0, 255),  # Màu đỏ
            thickness=-1
        )
    output_filename = f"captcha_{captcha_type}_result.png"
    cv2.imwrite(output_filename, result_image)
    print(f"✅ Đã lưu kết quả vào file: {output_filename}")


def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return text


def main():
    """
    Hàm chính để giải CAPTCHA từ ảnh chụp màn hình.
    Quy trình:
    1. Tải cấu hình và bản dịch.
    2. Đọc ảnh chụp màn hình.
    3. Tải mô hình YOLO.
    4. Phát hiện và cắt vùng CAPTCHA.
    5. Nhận dạng văn bản hướng dẫn.
    6. Xác định loại CAPTCHA (3x3/4x4).
    7. Giải CAPTCHA và hiển thị kết quả.
    """
    # Bước 1: Tải cấu hình và kiểm tra file bản dịch
    config = load_config('config/settings.yaml')
    translations_path = config['data']['translations']
    if not os.path.exists(translations_path):
        print(f"❌ Lỗi: Không tìm thấy file bản dịch tại '{translations_path}'")
        sys.exit(1)
    translations = load_translations(translations_path)

    # Bước 2: Xác định thiết bị (GPU hoặc CPU)
    device = get_device(config['device']['auto_detect'])

    # Bước 3: Đọc ảnh chụp màn hình
    screenshot_path = config['data']['screenshot']
    result_image = load_screenshot(screenshot_path)

    # Bước 4: Tải các mô hình YOLO
    detect_model = YOLO(config['models']['detect']).to(device)
    segment_model = YOLO(config['models']['segment']).to(device)
    classify_model = YOLO(config['models']['classify']).to(device)

    # Bước 5: Phát hiện vùng CAPTCHA chính
    captcha_region_x, captcha_region_y, cropped_captcha = detect_and_crop_captcha_region(
        open(screenshot_path, "rb").read(), detect_model)

    # Bước 6: Cắt vùng CAPTCHA và văn bản hướng dẫn
    offset_x, offset_y, captcha_image, text_image = extract_captcha_and_text(cropped_captcha)

    # Bước 7: Nhận dạng văn bản hướng dẫn
    detected_language, instruction_text = extract_instruction_text(text_image, use_gpu=(device == 'cuda'))
    print("Văn bản nhận dạng:", instruction_text)
    instruction_text = remove_accents(instruction_text.replace(" ","").lower())
    # Bước 8: Xác định loại CAPTCHA
    captcha_type = "3x3"
    text_indicators_4x4 = ["squareswith", "vuongco"]
    for indicator in text_indicators_4x4:
        if indicator in instruction_text:
            captcha_type = "4x4"
            break
    print("Loại CAPTCHA:", captcha_type)

    # Bước 9: Tính offset tổng để điều chỉnh tọa độ về ảnh gốc
    total_offset_x = captcha_region_x + offset_x
    total_offset_y = captcha_region_y + offset_y

    # Bước 10: Giải CAPTCHA và hiển thị kết quả
    if captcha_type == "3x3":
        cell_centers = solve_3x3_captcha(
            classify_model, captcha_image, instruction_text, detected_language,
            total_offset_x, total_offset_y, translations
        )
    else:
        cell_centers = solve_4x4_captcha(
            segment_model, captcha_image, instruction_text, detected_language,
            total_offset_x, total_offset_y, result_image, translations
        )
        print("Tọa độ tâm các ô được chọn:", cell_centers)

    draw_and_display_results(result_image, cell_centers, captcha_type)


if __name__ == "__main__":
    main()
