import os
import sys
import cv2
import numpy as np
from sympy import im

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..', 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.cv import deskew_image, draw_text_in_center, find_Lines, remove_regions, preProcessing
from utils.ocr import detectText
from utils.point import is_same_column, is_same_row, split_rows_columns, getColumnIndex, getRowIndex, is_have_line
from utils.table import cells_to_html, createHTML

def findCenters(img_bin):
    """
    Tìm trọng tâm của các bbox trong ảnh. Thực hiện bồi đắp theo kernel 3x3, sao đó tìm bbox và tính center.

    Tham số:
    - img_bin: Ảnh nhị phân.

    Trả về:
    - centers: Danh sách các tọa độ trọng tâm của các bbox.
    """
    kernel = np.ones((3,3),np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        centers.append((x + w // 2, y + h // 2))
    return centers

def create_cells(rows, columns, tableMask):
    """
    Tạo các ô bảng từ các điểm giao đã được phân theo hàng và cột.

    Tham số:
    - rows: Danh sách các hàng.
    - columns: Danh sách các cột.

    Trả về:
    - cells: Danh sách các ô bảng.
    Với mỗi ô bảng, lưu trữ các thông tin sau:
        - bbox: Bounding box của ô bảng.
        - row_span: Số hàng của ô bảng.
        - col_span: Số cột của ô bảng.
        - row: Vị trí hàng của ô bảng.
        - col: Vị trí cột của ô bảng.
    """
    cells = []

    for row_index in range(len(rows)-1):
        for top_left_index in range(len(rows[row_index]) - 1):
            top_left = rows[row_index][top_left_index]
            top_right = rows[row_index][top_left_index + 1]
            if not is_have_line(top_left, top_right, tableMask): continue
            # Tìm cột của top_left
            top_left_col_index = getColumnIndex(top_left, columns)
            # duyệt qua cột vừa tìm được và tìm bottom_left
            lower_left = list(filter(lambda c: c[1] > top_left[1], columns[top_left_col_index]))
            if len(lower_left) == 0: continue
            # bottom_left = lower_left[0]
            isCorrectBottomLeft = False
            for bottom_left in lower_left:
                if not is_have_line(top_left, bottom_left, tableMask): break
                for top_right_index in range(top_left_index + 1, len(rows[row_index])):
                    top_right = rows[row_index][top_right_index]
                    if not is_have_line(top_left, top_right, tableMask): continue
                    # if not is_have_line(top_left, top_right, tableMask): continue
                    # Tìm cột của top_right
                    top_right_col_index = getColumnIndex(top_right, columns)
                    # duyệt qua cột vừa tìm được và tìm bottom_left
                    lower_right = list(filter(lambda c: c[1] > top_right[1], columns[top_right_col_index]))
                    if len(lower_right) == 0: continue
                    isCorrectBottomRight = False
                    for bottom_right in lower_right:
                        if not is_same_row(bottom_left, bottom_right): continue
                        if not is_have_line(bottom_left, bottom_right, tableMask): continue
                        if not is_have_line(top_right, bottom_right, tableMask): continue
                        isCorrectBottomRight = True
                        isCorrectBottomLeft = True
                        # Tìm hàng của bottom_right
                        bottom_right_row_index = getRowIndex(bottom_right, rows)
                        # Tạo bbox
                        bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                        # tính row_span và col_span của bbox
                        row_span = bottom_right_row_index - row_index
                        col_span = top_right_col_index - top_left_col_index
                        # Tạo cell
                        cell = {'bbox': bbox, 'row_span': row_span, 'col_span': col_span, 'row':row_index, 'col':top_left_col_index}
                        cells.append(cell)
                        break
                    if isCorrectBottomRight: break
                if isCorrectBottomLeft: break
    return cells


def draw_cells(img, cells, size = 0.7, color = (0, 0, 255)):
    img = np.zeros_like(img)
    for index, cell in enumerate(cells, start=1):
        x1, y1, x2, y2 = cell['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 60, 255), 1)
        draw_text_in_center(img, f"{cell['row_span']}x{cell['col_span']}", cell['bbox'], size, color)
        # center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # cv2.circle(img, (center), 5, (255, 0, 255), -1)
    return img


def recognize(image_path, detector, useBase64=False):
    cells, image_removed = imgPath_to_cells(image_path)
    cells = add_text_to_cell(cells, image_removed, detector)
    html = cells_to_html(cells).replace('<thead>','<tr>').replace('</thead>','</tr>').replace('\n',"<br>")
    new_html = createHTML(image_path, html, show_image=True, useBase64=useBase64)
    return new_html
    
def add_text_to_cell(cells, image_removed, detector):
    cells_imgs = []
    for cell in cells:
        x1,y1,x2,y2 = cell['bbox']
        cropped_image = image_removed[int(y1):int(y2), int(x1):int(x2)]
        img = cropped_image #trim_white(cropped_image)
        cells_imgs += [(img, cell['bbox'])]
    texts = []
    for cell in cells_imgs:
        img, bbox = cell
        text, lines = detectText(img, detector)
        texts += [text]
    cells_imgs = []
    for i in range(len(cells)):
        cells[i]['cell text'] = texts[i]
    return cells

def imgPath_to_cells(image_path):
    image = cv2.imread(image_path)
    return img_to_cells(image)

def img_to_cells(image):
    image_ok, calc_angle = deskew_image(image)
    _, image_pre, _ = preProcessing(image_ok)
    mask, dots, outImag = find_Lines(image_pre)
    image_removed = remove_regions(image_pre, mask)
    centers = findCenters(dots)
    rows = split_rows_columns(centers, modeName='row')
    columns = split_rows_columns(centers, modeName='column')
    cells = create_cells(rows, columns, mask)
    return cells, image_removed