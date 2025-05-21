from ultralytics import YOLO
import numpy as np
import cv2
from .utilities import get_translation


def solve_3x3_captcha(
        model: YOLO,
        image: np.ndarray,
        instruction_text: str,
        language: str,
        offset_x: int,
        offset_y: int,
        translations: dict
) -> list:
    """
    Giải CAPTCHA 3x3 bằng cách chia lưới, phân loại ô, và chọn ô khớp văn bản.

    Args:
        model (YOLO): Mô hình YOLO để phân loại ô.
        image (np.ndarray): Ảnh CAPTCHA (BGR).
        instruction_text (str): Văn bản hướng dẫn.
        language (str): Ngôn ngữ nhận dạng ("vi" hoặc "en").
        offset_x (int): Offset tọa độ x để điều chỉnh về ảnh gốc.
        offset_y (int): Offset tọa độ y để điều chỉnh về ảnh gốc.
        translations (dict): Từ điển bản dịch các lớp.

    Returns:
        list: Danh sách tọa độ tâm các ô được chọn [(x, y), ...].
    """
    if image is None:
        print("❌ Lỗi: Ảnh CAPTCHA không hợp lệ.")
        return []

    # Chia ảnh thành lưới 3x3
    height, width = image.shape[:2]
    cell_height = height // 3
    cell_width = width // 3
    cell_images = []

    for row in range(3):
        for col in range(3):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            cell_image = image[y1:y2, x1:x2]
            cell_images.append(cell_image)

    # Phân loại từng ô và tìm ô khớp văn bản
    cell_scores = []
    for idx, cell_image in enumerate(cell_images):
        results = model(cell_image)
        if results and results[0].probs:
            class_id = int(results[0].probs.top1)
            class_name = model.names[class_id]
            confidence = float(results[0].probs.top1conf)
            translated_name = get_translation(translations, class_name, language)
            if translated_name in instruction_text:
                cell_scores.append((idx, confidence))
                print(f"Ô {idx}: lớp={class_name}, độ tin cậy={confidence:.4f}")

    # Sắp xếp và chọn top 3 ô
    cell_scores.sort(key=lambda x: x[1], reverse=True)
    top_3_indices = [idx for idx, _ in cell_scores[:3]]
    print("Top 3 ô được chọn:", top_3_indices)

    # Tính tọa độ tâm các ô
    cell_centers = []
    for idx in top_3_indices:
        row = idx // 3
        col = idx % 3
        center_x = col * cell_width + cell_width // 2 + offset_x
        center_y = row * cell_height + cell_height // 2 + offset_y
        cell_centers.append((center_x, center_y))

    return cell_centers


def solve_4x4_captcha(
        model: YOLO,
        image: np.ndarray,
        instruction_text: str,
        language: str,
        offset_x: int,
        offset_y: int,
        result_image: np.ndarray,
        translations: dict
) -> list:
    """
    Giải CAPTCHA 4x4 bằng cách tiền xử lý ảnh (xóa đường kẻ trắng), phân đoạn,
    và xác định ô chứa đối tượng khớp văn bản.

    Args:
        model (YOLO): Mô hình YOLO để phân đoạn.
        image (np.ndarray): Ảnh CAPTCHA (BGR).
        instruction_text (str): Văn bản hướng dẫn.
        language (str): Ngôn ngữ nhận dạng ("vi" hoặc "en").
        offset_x (int): Offset tọa độ x để điều chỉnh về ảnh gốc.
        offset_y (int): Offset tọa độ y để điều chỉnh về ảnh gốc.
        result_image (np.ndarray): Ảnh gốc (không dùng, để tương thích).
        translations (dict): Từ điển bản dịch các lớp.

    Returns:
        list: Danh sách tọa độ tâm các ô được chọn [(x, y), ...].
    """
    if image is None:
        print("❌ Lỗi: Ảnh CAPTCHA không hợp lệ.")
        return []

    # Tiền xử lý: Xóa đường kẻ trắng
    height, width = image.shape[:2]
    processed_image = image.copy()

    # Chuyển sang grayscale và tạo mask cho vùng trắng
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # Ngưỡng gần trắng
    white_mask = cv2.dilate(white_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Nội suy để xóa đường kẻ trắng
    processed_image = cv2.inpaint(image, white_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Chia ảnh thành lưới 4x4
    grid_size = 4
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Lưu ảnh đã xử lý để debug (tùy chọn)
    cv2.imwrite(f"processed_captcha_{language}.png", processed_image)

    # Chạy phân đoạn trên ảnh đã xử lý
    results = model(processed_image, conf=0.5)
    if not results or results[0].masks is None or results[0].boxes is None:
        print("❌ Lỗi: Không phát hiện được đối tượng với mask.")
        return []

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    names = model.names

    occupied_cells = set()
    cell_centers = []

    print(f"Phát hiện {len(masks)} đối tượng với mask.")
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        class_name = names[int(box.cls.cpu().numpy())]
        translated_name = get_translation(translations, class_name, language)
        print(f"Đối tượng: {class_name}, Bản dịch: {translated_name}")

        if translated_name not in instruction_text:
            continue

        print(f"Xử lý mask cho {class_name}")
        # Chuyển mask thành ảnh nhị phân
        mask_binary = (mask * 255).astype(np.uint8)
        resized_mask = cv2.resize(mask_binary, (width, height), interpolation=cv2.INTER_NEAREST)
        coords = np.column_stack(np.where(resized_mask > 0))

        if len(coords) == 0:
            print("Không có điểm nào trong mask.")
            continue

        # Tính tâm của mask
        center_x = coords[:, 1].mean()
        center_y = coords[:, 0].mean()

        # Thu nhỏ 2% về phía tâm
        shrink_factor = 0.98
        shrunk_coords = []
        for y_coord, x_coord in coords:
            dx = x_coord - center_x
            dy = y_coord - center_y
            new_x = int(center_x + dx * shrink_factor)
            new_y = int(center_y + dy * shrink_factor)
            shrunk_coords.append((new_y, new_x))

        # Xác định ô lưới chứa các điểm
        for y_coord, x_coord in shrunk_coords:
            row = int(y_coord // cell_height)
            col = int(x_coord // cell_width)
            row = max(0, min(grid_size - 1, row))
            col = max(0, min(grid_size - 1, col))
            cell_index = row * grid_size + col
            occupied_cells.add(cell_index)

    # Tính tâm ô để vẽ
    for cell_index in occupied_cells:
        row = cell_index // grid_size
        col = cell_index % grid_size
        center_x = col * cell_width + cell_width // 2 + offset_x
        center_y = row * cell_height + cell_height // 2 + offset_y
        cell_centers.append((center_x, center_y))

    if not occupied_cells:
        print("❌ Lỗi: Không tìm thấy ô nào khớp văn bản.")
        return []

    print("Các ô được chọn (chỉ số 1D):", list(occupied_cells))
    return cell_centers
