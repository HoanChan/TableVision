import os
import sys
import cv2
import numpy as np
from sympy import im

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..', 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.cv import deskew_image, draw_text_in_center, remove_regions, preProcessing
from utils.ocr import detectText
from utils.point import split_rows_columns
from utils.table import cells_to_html, createHTML

def detect_lines(img_bin, fixkernel, detectkernel):
    """
    Phát hiện các đường thẳng trong ảnh bằng cách sử dụng các kernel được chỉ định.
    Cụ thể, ảnh được lấp đầy các khoảng trống với fixkernel.
    Sau đó loại bỏ các thành phần không phải đường thẳng với detectkernel.
    Cuối cùng là phục hồi lại như cũ với detectkernel.

    Tham số:
    - img_bin: Ảnh nhị phân.
    - fixkernel: Kernel được sử dụng để dilate (Bồi đắp các pixel màu trắng) để sửa lỗi.
    - detectkernel: Kernel được sử dụng để erode (Bào mòn các pixel màu trắng) và dilate nhằm giữ lại các đường.

    Kết quả:
    - result: Ảnh sau khi phát hiện các đường thẳng.
    """
    image_0 = cv2.dilate(img_bin, fixkernel, iterations=2)
    image_1 = cv2.erode(image_0, detectkernel, iterations=3)
    result = cv2.dilate(image_1, detectkernel, iterations=3)
    return result

def find_Lines(img):
    """
    Nhận diện cấu trúc bảng trong ảnh.

    Tham số:
    - img: Ảnh cần nhận diện cấu trúc.

    Kết quả:
    - img_vh: Ảnh chứa thông tin về các đường kẻ dọc và ngang (mask).
    - img_sub: Ảnh chứa thông tin các điểm giao nhau của các đường dọc và ngang.
    - outImag: Danh sách các ảnh trung gian.
    """    
    outImag=[]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    # THRESH_OTSU là phương pháp tự động xác định ngưỡng dựa trên histogram của ảnh
    thresh, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    outImag.append((img_bin, 'invert'))

    # Giả sử bảng có tối đa 50 dòng và 50 cột
    kernel_len_ver = img_height // 50
    kernel_len_hor = img_width // 50
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    vertical_lines = detect_lines(img_bin, kernel, ver_kernel)
    outImag.append((vertical_lines, 'vertical_lines'))

    horizontal_lines = detect_lines(img_bin, kernel, hor_kernel)
    outImag.append((horizontal_lines, 'horizontal_lines'))

    img_vh = cv2.bitwise_or(vertical_lines, horizontal_lines)
    outImag.append((img_vh, 'Combine'))

    img_vh = cv2.dilate(img_vh, kernel, iterations=1)
    outImag.append((img_vh, 'Dilated - Mask Result'))

    img_sub = cv2.bitwise_and(vertical_lines, horizontal_lines)
    outImag.append((img_sub, 'Subtraction - Point Result'))

    return img_vh, img_sub, outImag


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


def create_cells(rows, columns):
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
            # Tìm cột của top_left
            top_left_col_index = 0
            for top_left_col_index in range(len(columns)):
                if abs(columns[top_left_col_index][0][0] - top_left[0]) < 10:
                    break
            # duyệt qua cột vừa tìm được và tìm bottom_left
            lower_left = list(filter(lambda c: c[1] > top_left[1], columns[top_left_col_index]))
            if len(lower_left) == 0: continue
            bottom_left = lower_left[0]
            for top_right_index in range(top_left_index + 1, len(rows[row_index])):
                top_right = rows[row_index][top_right_index]
                # Tìm cột của top_right
                top_right_col_index = 0
                for top_right_col_index in range(len(columns)):
                    if abs(columns[top_right_col_index][0][0] - top_right[0]) < 10:
                        break
                # duyệt qua cột vừa tìm được và tìm bottom_left
                lower_right = list(filter(lambda c: c[1] > top_right[1], columns[top_right_col_index]))
                if len(lower_right) == 0: continue
                bottom_right = lower_right[0]
                
                # Xử lý bottom_left và bottom_right không cùng 1 hàng
                bottom_right = (bottom_right[0], max(bottom_left[1], bottom_right[1]))
                bottom_left = (bottom_left[0], max(bottom_left[1], bottom_right[1]))
                # print('top_left',top_left,'top_right', top_right, 'bottom_right', bottom_right)

                # Tìm hàng của bottom_right
                bottom_right_row_index = 0
                for bottom_right_row_index in range(len(rows)):
                    if abs(rows[bottom_right_row_index][0][1] - bottom_right[1]) < 10:
                        break

                # Tạo bbox
                bbox =  (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                # print(bbox)
                # tính row_span và col_span của bbox
                row_span = bottom_right_row_index - row_index
                col_span = top_right_col_index - top_left_col_index
                # Tạo cell
                cell = {'bbox': bbox, 'row_span': row_span, 'col_span': col_span, 'row':row_index, 'col':top_left_col_index}
                cells.append(cell)
                break
                # return cells
    return cells


def draw_cells(img, cells, size = 0.7, color = (0, 0, 255)):
    img = img.copy()
    for index, cell in enumerate(cells, start=1):
        x1, y1, x2, y2 = cell['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        draw_text_in_center(img, f"{cell['row_span']}x{cell['col_span']}", cell['bbox'], size, color)
        # cv2.circle(img, center, 5, (255, 0, 255), -1)
    return img


def recognize(image_path, detector, useBase64=False):
    image = cv2.imread(image_path)
    _, image = preProcessing(image)
    image_ok, calc_angle = deskew_image(image)
    mask, dots, outImag = find_Lines(image_ok)
    image_removed = remove_regions(image_ok, mask)
    centers = findCenters(dots)
    rows = split_rows_columns(centers, mode='row')
    columns = split_rows_columns(centers, mode='column')
    cells = create_cells(rows, columns)
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
    html = cells_to_html(cells).replace('<thead>','<tr>').replace('</thead>','</tr>').replace('\n',"<br>")
    new_html = createHTML(image_path, html, show_image=True, useBase64=useBase64)
    return new_html