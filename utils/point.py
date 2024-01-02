import numpy as np
import cv2

from typing import List, Tuple
Point = Tuple[int, int]
def split_rows_columns(centers: List[Point], modeName: str ='row') -> List[List[Point]]:
    mode = 0 if modeName == 'row' else 1
    centers.sort(key=lambda c: c[1-mode])  # Sắp xếp centers theo tọa độ y hoặc x tùy theo mode
    rows_columns = []
    current_row_column = [centers[0]]
    for center in centers[1:]:
        if abs(center[1-mode] - current_row_column[0][1-mode]) <= 10:  # Điểm y hoặc x gần nhau thuộc cùng một hàng hoặc cột
            current_row_column.append(center)
        else:  # Điểm y hoặc x xa, tạo hàng hoặc cột mới
            rows_columns.append(current_row_column)
            current_row_column = [center]
    rows_columns.append(current_row_column)  # Thêm hàng/cột cuối cùng
    # Sắp xếp lại các center trong mỗi hàng/cột theo tọa độ x hoặc y tùy theo mode
    for row_column in rows_columns:
        row_column.sort(key=lambda c: c[mode])
    return rows_columns

def is_same_row(point1: Point, point2: Point, threshold: int = 10) -> bool:
    return abs(point1[1] - point2[1]) <= threshold

def is_same_column(point1: Point, point2: Point, threshold: int = 10) -> bool:
    return abs(point1[0] - point2[0]) <= threshold

def getColumnIndex(point: Point, columns: List[List[Point]]) -> int:
    index = 0
    for index in range(len(columns)):
        if is_same_column(columns[index][0], point):
            break
    return index

def getRowIndex(point: Point, rows: List[List[Point]]) -> int:
    index = 0
    for index in range(len(rows)):
        if is_same_row(rows[index][0], point):
            break
    return index

def is_have_line(point1: Point, point2: Point, mask:np.ndarray, threshold: float = 0.5) -> bool:
    line_points = cv2.line(np.zeros_like(mask), point1, point2, color=(255, 255, 255), thickness=1)
    # tạo 1 ảnh mới bằng cách chỉ giữ lại các điểm trên mask mà cũng điểm đó ở line_points có màu trắng
    new_mask = mask.copy()
    new_mask[line_points == 0] = 0
    # tính tỉ lệ số điểm trắng trên tổng số điểm của line_points
    ratio = np.sum(new_mask) / np.sum(line_points)
    return ratio >= threshold