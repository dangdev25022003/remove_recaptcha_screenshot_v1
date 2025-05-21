from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

def detect_and_crop_captcha_region(
    screenshot_bytes: bytes,
    model: YOLO,
    show: bool = False
) -> tuple:
    """
    Phát hiện và cắt vùng CAPTCHA chính từ ảnh chụp màn hình.

    Args:
        screenshot_bytes (bytes): Dữ liệu bytes của ảnh chụp màn hình.
        model (YOLO): Mô hình YOLO để phát hiện vùng CAPTCHA.
        show (bool): Nếu True, hiển thị kết quả phát hiện.

    Returns:
        tuple: (x, y, cropped_image) - Tọa độ góc trên-trái và ảnh cắt.
               Trả về (None, None, None) nếu không phát hiện được.
    """
    # Chuyển bytes thành ảnh PIL
    try:
        pil_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    except Exception as e:
        print(f"❌ Lỗi khi mở ảnh: {e}")
        return None, None, None

    # Chuyển PIL thành OpenCV (BGR)
    image_array = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Chạy mô hình YOLO để phát hiện vùng CAPTCHA
    results = model(pil_image)

    if not results or len(results[0].boxes) == 0:
        print("❌ Lỗi: Không phát hiện được CAPTCHA.")
        return None, None, None

    if show:
        results[0].show()

    # Lấy vùng có độ tin cậy cao nhất
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    max_conf_idx = confidences.argmax()
    x1, y1, x2, y2 = map(int, boxes[max_conf_idx])

    # Cắt vùng CAPTCHA
    cropped_image = image_bgr[y1:y2, x1:x2]
    return x1, y1, cropped_image