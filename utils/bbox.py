import numpy as np
from utils.point import split_rows_columns

def is_bbox_overlap(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    if x2_2 < x1_1: return False # Kiểm tra xem bbox2 có nằm bên trái bbox1 hay không
    if x1_2 > x2_1: return False # Kiểm tra xem bbox2 có nằm bên phải bbox1 hay không
    if y2_2 < y1_1: return False # Kiểm tra xem bbox2 có nằm phía trên bbox1 hay không
    if y1_2 > y2_1: return False # Kiểm tra xem bbox2 có nằm phía dưới bbox1 hay không
    return True # Nếu không thoả mãn các điều kiện trên, hai bbox chồng lên nhau

def is_bboxs_overlap(box, bboxs):
    for bbox in bboxs:
        if is_bbox_overlap(box, bbox):
            return True
    return False

def isSameRow(bbox1, bbox2, overlap_percent = 50 ):
    min_height = min(bbox1[3] - bbox1[1], bbox2[3] - bbox2[1])
    overlap_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    return overlap_height / min_height * 100 >= overlap_percent

def isSameCol(bbox1, bbox2, overlap_percent = 50):
    min_width = min(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
    overlap_width = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    return overlap_width / min_width * 100 >= overlap_percent

def is_bbox_inside(bbox1, bbox2):
    # Kiểm tra xem bbox1 có nằm trong bbox2 hay không
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2

def is_bbox_outside(bbox1, bbox2):
    return not is_bbox_overlap(bbox2, bbox1)

def merge_bbox(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return (x1, y1, x2, y2)

def create_bbox(boxs):
    x1 = min([box[0] for box in boxs])
    y1 = min([box[1] for box in boxs])
    x2 = max([box[2] for box in boxs])
    y2 = max([box[3] for box in boxs])
    return (x1, y1, x2, y2)

def get_same_row_col_bboxs(bbox, bboxs, mode = 'row', overlap_percent = 0):
    # Tìm tất cả các bbox cùng hàng với bbox sao cho bbox bao toàn bộ chúng không chồng lên bbox khác
    boxs = []
    others = []
    isSame = isSameRow if mode == 'row' else isSameCol
    for box in bboxs:
        if isSame(bbox, box, overlap_percent):
            boxs.append(box)
        else:
            others.append(box)
    bbox = create_bbox(boxs)
    # Lọc ra các bbox không cùng cột nhưng lại chồng lên bbox
    others = [box for box in others if is_bbox_overlap(bbox, box)]
    # duyệt qua others
    for box in others:
        # duyệt qua boxs theo thứ tự ngược
        for i in range(len(boxs) - 1, -1, -1):
            if isSame(box, boxs[i], overlap_percent):
                boxs.pop(i)
    return boxs

def create_mask_from_bboxs(image, bboxs):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    return mask

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def split_box_rows_columns(bboxs, mode='row'):
    mode = 0 if mode == 'row' else 1
    bboxs.sort(key=lambda c: c[1-mode])  # Sắp xếp bboxs theo tọa độ y hoặc x tùy theo mode
    rows_columns = []
    current_row_column = [bboxs[0]]
    for box in bboxs[1:]:
        if (mode ==0 and isSameRow(box,current_row_column[0])) or (mode !=0 and isSameCol(box, current_row_column[0])):  # Điểm y hoặc x gần nhau thuộc cùng một hàng hoặc cột
            current_row_column.append(box)
        else:  # Điểm y hoặc x xa, tạo hàng hoặc cột mới
            rows_columns.append(current_row_column)
            current_row_column = [box]
    rows_columns.append(current_row_column)  # Thêm hàng/cột cuối cùng

    # Xử lý trường hợp rowspan/colspan 
    for row_column in rows_columns:
        tops = [box[:2] for box in row_column]
        cols = split_rows_columns(tops, 'row' if mode == 0 else 'col')
        # thay thế row_column bằng các phần tử của cols
        index = rows_columns.index(row_column)
        # lập danh sách các box dựa vào tops
        box_cols = [[box for box in row_column if box[:2] in col] for col in cols]
        if len(box_cols) > 1:
            # xóa row_column hiện tại
            rows_columns = rows_columns[:index] + box_cols + rows_columns[index+1:]
       
    # Sắp xếp lại các box trong mỗi hàng/cột theo tọa độ x hoặc y tùy theo mode
    for row_column in rows_columns:
        row_column.sort(key=lambda c: c[1-mode])
    return rows_columns

def get_scale_bbox(box, scale):
    return (int(box[0] * scale), int(box[1] * scale), int(box[2] * scale), int(box[3] * scale)) 
