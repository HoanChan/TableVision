from ..utils.bbox import *
from ..utils.cv import *
from ..utils.math import *

def explain_bboxs_by_width(bboxs, image, row_threshold = 20):
    width, height = image.shape[:2]
    # duyệt từng bbox, mở rộng bbox theo chiều ngang cho tới khi sắp chạm vào bbox bên cạnh hoặc biên
    for index in range(len(bboxs)):
        bbox = bboxs[index]
        bbox_width = bbox[2] - bbox[0]
        # Lập danh sách các bbox cùng cột với nó
        same_column_bboxs = get_same_row_col_bboxs(bbox, bboxs, mode='col', overlap_percent=0)
        # ratio_threshold = 2
        # same_column_bboxs = [box for box in same_column_bboxs if (box[2] - box[0]) / bbox_width  <= ratio_threshold]
        x1_min = np.min([box[0] for box in same_column_bboxs])
        x2_max = np.max([box[2] for box in same_column_bboxs])        
        # thay đổi x1 và x2 của bbox hiện tại sao cho nằm trong cột và không chạm vào bbox bên cạnh
        x2_max_left = max_with_default([box[2] for box in bboxs if box[2] + row_threshold < x1_min], x1_min)
        x1_min = max(x1_min, x2_max_left)
        
        x1_min_right = min_with_default([box[0] for box in bboxs if box[0] > x2_max + row_threshold], x2_max)
        x2_max = min(x2_max, x1_min_right)

        # cập nhật lại bbox
        bboxs[index] = (x1_min, bbox[1], x2_max, bbox[3])
    return bboxs

def explain_bboxs_by_column(bboxs, image, row_threshold = 20):
    width, height = image.shape[:2]
    # duyệt từng bbox, mở rộng bbox theo chiều ngang theo cột
    checker = [False] * len(bboxs)
    for index in range(len(bboxs)):
        if checker[index]: continue
        bbox = bboxs[index]
        x1_min = bbox[0]
        x2_max = bbox[2]
        # Mở rộng bbox sang trái
        x2_max_left = max_with_default([box[2] for box in bboxs if box[2] + row_threshold//2 < x1_min])
        if x2_max_left == 0 :
            x1_min = row_threshold
        elif x2_max_left < x1_min - row_threshold:
            x1_min = (x1_min + x2_max_left)//2
           
        # Mở rộng bbox sang phải
        x1_min_right = min_with_default([box for box in bboxs if box[0] > x2_max + row_threshold//2])
        if x1_min_right == 0:
            x2_max = width - row_threshold
        else:
            x2_max = max(x2_max, x1_min_right - row_threshold)
        
        # cập nhật lại bbox
        same_column_bboxs = [(i, box) for i, box in enumerate(bboxs) if isSameCol(bbox, box) and checker[i] == False]
        for i, box in same_column_bboxs:
            bboxs[i] = (x1_min, box[1], x2_max, box[3])
            checker[i] = True
    return bboxs

def explain_bboxs_by_space(bboxs, image, row_threshold = 20, col_threshold = 20):
    width, height = image.shape[:2]
    # tìm tất cả bbox mà không có bbox nào ở bên phải và cùng hàng với nó rồi mở rộng nó, nếu có thể thì mở rộng về cả 2 phía luôn
    no_right_bboxs = [box for box in bboxs if len([box2 for box2 in bboxs if isSameRow(box, box2, overlap_percent=0) and box2[0] > box[2]]) == 0]
    for index in range(len(bboxs)):
        if bboxs[index] in no_right_bboxs:
            x1, y1, x2, y2 = bboxs[index]
            explain_right_width = width - x2 - row_threshold            
            new_box = (max(row_threshold, x1 - explain_right_width), y1, x2 + explain_right_width, y2)           
            # Điều chỉnh lại x1 cho đúng cột
            same_column_bboxs = [box for box in bboxs if isSameCol(new_box, box, overlap_percent=0) and box != bboxs[index]]
            new_box = (min_with_default([box[0] for box in same_column_bboxs], new_box[0]), y1, x2 + explain_right_width, y2)
            same_row_bboxs = [box for box in bboxs if isSameRow(bboxs[index], box, overlap_percent=0) and box != bboxs[index]]          
            isOverlap = is_bboxs_overlap(new_box, same_row_bboxs)
            if not isOverlap:
                bboxs[index] = create_bbox([bboxs[index], new_box])
            else:
                bboxs[index] = create_bbox([bboxs[index], (x1, y1, width - row_threshold, y2)])
    # tìm tất cả các bbox có các bbox cùng hàng nằm cách xa nó 1 khoảng lớn hơn ngưỡng rồi mở rộng nó
    right_bboxs = [(box, min_with_default([box2[0] for box2 in bboxs if isSameRow(box, box2) and box2[0] > box[2]])) for box in bboxs] 
    far_right_bboxs = [(box, min) for box, min in right_bboxs if min > box[2] + row_threshold * 2]
    for index in range(len(bboxs)):
        for far_right_bbox, min in far_right_bboxs:
            if bboxs[index] == far_right_bbox:
                bboxs[index] = create_bbox([bboxs[index], (bboxs[index][0], bboxs[index][1], min - row_threshold, bboxs[index][3])])
    # tìm tất cả các bbox mà không có bbox nào ở bên dưới và cùng cột với nó rồi mở rộng nó
    no_bottom_bboxs = [box for box in bboxs if len([box2 for box2 in bboxs if isSameCol(box, box2) and box2[1] > box[3]]) == 0]
    for index in range(len(bboxs)):
        if bboxs[index] in no_bottom_bboxs:
            bboxs[index] = create_bbox([bboxs[index], (bboxs[index][0], bboxs[index][1], bboxs[index][2], height - col_threshold*2)])
    # tìm tất cả các bbox có các bbox cùng cột nằm cách xa nó 1 khoảng lớn hơn ngưỡng rồi mở rộng nó
    bottom_bboxs = [(box, min_with_default([box2[1] for box2 in bboxs if isSameCol(box, box2) and box2[1] > box[3]])) for box in bboxs]
    far_bottom_bboxs = [(box, min) for box, min in bottom_bboxs if min > box[3] + col_threshold * 2]
    for index in range(len(bboxs)):
        for far_bottom_bbox, min in far_bottom_bboxs:
            if bboxs[index] == far_bottom_bbox:
                bboxs[index] = create_bbox([bboxs[index], (bboxs[index][0], bboxs[index][1], bboxs[index][2], min - col_threshold)])
    # tìm tất cả các bbox không có bbox nào ở trên nó và cùng cột với nó rồi mở rộng nó
    no_top_bboxs = [box for box in bboxs if len([box2 for box2 in bboxs if isSameCol(box, box2) and box2[3] < box[1]]) == 0]
    for index in range(len(bboxs)):
        if bboxs[index] in no_top_bboxs:
            bboxs[index] = create_bbox([bboxs[index], (bboxs[index][0], col_threshold, bboxs[index][2], bboxs[index][3])])
    return bboxs

def normalize_bboxs(bboxs, row_threshold = 10, col_threshold = 10):
    # Duyệt qua từng bbox
    for index in range(len(bboxs)):
        # tìm tất cả các bbox cùng hàng với nó
        min_x1 = np.min([box[0] for box in bboxs if abs(bboxs[index][0] - box[0]) < row_threshold])
        min_y1 = np.min([box[1] for box in bboxs if abs(bboxs[index][1] - box[1]) < col_threshold])
        max_x2 = np.max([box[2] for box in bboxs if abs(box[2] - bboxs[index][2]) < row_threshold])
        max_y2 = np.max([box[3] for box in bboxs if abs(box[3] - bboxs[index][3]) < col_threshold])
        bboxs[index] = (min_x1, min_y1, max_x2, max_y2)
    return bboxs

def find_Cells(image):    
    """
    Nhận diện cấu trúc bảng trong ảnh.

    Tham số:
    - image: Ảnh cần nhận diện cấu trúc.

    Kết quả:
    - img_vh: Ảnh chứa thông tin về các đường kẻ dọc và ngang (mask).
    - img_sub: Ảnh chứa thông tin các điểm giao nhau của các đường dọc và ngang.
    - outImag: Danh sách các ảnh trung gian.
    """    
    outImag=[]
    # outImag.append((image, 'image'))
    img_height, img_width = image.shape[:2]
    if img_height == 0 or img_width == 0:
        return [], [], image, outImag
    # THRESH_OTSU là phương pháp tự động xác định ngưỡng dựa trên histogram của ảnh
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # outImag.append((img_bin, 'invert'))
    try:
        img_median = cv2.medianBlur(img_bin, 11)
    except:
        img_median = img_bin
    outImag.append((img_median, 'img_median'))

    
    # Connect letters that are connected only by a few pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    img_connect = cv2.dilate(img_median, kernel, iterations=3)
    outImag.append((img_connect, 'connect_letters'))

    # loại bỏ nhiễu
    img_bold = cv2.medianBlur(img_connect, 9)
    outImag.append((img_bold, 'medianBlur'))

    # Xác định các contours
    contours, hierarchy = cv2.findContours(img_bold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sắp xếp các contours theo vị trí từ trên xuống dưới, từ trái qua phải
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1] * img_width + cv2.boundingRect(ctr)[0]) 

    bboxs = [cv2.boundingRect(contour) for contour in contours]
    # chuyển sang x1, y1, x2, y2
    bboxs = [(x, y, x+w, y+h) for x, y, w, h in bboxs]
    
    mask = create_mask_from_bboxs(img_bold, bboxs)
    outImag.append((draw_bboxs(mask, bboxs), 'mask'))
    
    bboxs = explain_bboxs_by_width(bboxs, mask)
    mask = create_mask_from_bboxs(img_bold, bboxs)
    outImag.append((draw_bboxs(mask, bboxs), 'explain by width'))

    # Loại bỏ các bbox chồng lấp nhau (loại bỏ bbox nhỏ hơn)
    small_bboxs = [(i,bbox) for i, bbox in enumerate(bboxs) if len([box for box in bboxs if is_bbox_overlap(bbox, box) and bbox != box and bbox_area(bbox) < bbox_area(box)]) > 0]
    bboxs = [bbox for i, bbox in enumerate(bboxs) if i not in [index for index, bbox in small_bboxs]]
    
    # bboxs = explain_bboxs_by_column(bboxs, mask)
    # mask = create_mask_from_bboxs(img_bold, bboxs)
    # outImag.append((draw_bboxs(mask, bboxs), 'explain by columns'))

    bboxs = explain_bboxs_by_space(bboxs, mask)
    mask = create_mask_from_bboxs(img_bold, bboxs)
    outImag.append((draw_bboxs(mask, bboxs), 'explain by space'))

    bboxs = normalize_bboxs(bboxs)
    mask = create_mask_from_bboxs(img_bold, bboxs)
    outImag.append((draw_bboxs(mask, bboxs), 'normalize bboxs'))
    
    # Sắp xếp các bboxs theo vị trí từ trên xuống dưới, từ trái qua phải
    bboxs = sorted(bboxs, key=lambda box: box[1] * img_width + box[0])
    
    return bboxs, mask, outImag

def getbox_index(box, rows, cols):
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            if box in row and box in col:
                return i, j
    return -1, -1

def add_missing_cells(bboxs, rows, cols, box_indexs):
    matrix = [[None for col in cols] for row in rows]
    for index, span in enumerate(box_indexs):
        matrix[span[0]][span[1]] = index
    # Duyệt qua toàn bộ matrix, nếu có ô nào là None thì thêm vào bboxs
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if cell is None:
                row_height = rows[i][-1][3] - rows[i][-1][1]
                col_width = cols[j][-1][2] - cols[j][-1][0]
                cel_x1, cel_y1 = cols[j][-1][0], rows[i][-1][1]
                cel_x2, cel_y2 = cel_x1 + col_width, cel_y1 + row_height
                # tạo bbox mới
                new_bbox = (cel_x1, cel_y1, cel_x2, cel_y2)
                # kiểm tra xem bbox mới có nằm chồng lên bbox cùng hàng hoặc cùng cột không dựa vào x1, y1, x2, y2 của nó                
                if is_bboxs_overlap(new_bbox, bboxs): continue
                bboxs.append(new_bbox)
                # cập nhật lại rows, cols, box_indexs
                rows[i].append(new_bbox)
                cols[j].append(new_bbox)
                box_indexs.append((i, j))
    #sắp xếp lại bboxs theo thứ tự từ trên xuống dưới, từ trái qua phải
    bboxs = sorted(bboxs, key=lambda box: box[1] * 1000000 + box[0])
    box_indexs = [getbox_index(box, rows, cols) for box in bboxs]
    return bboxs, rows, cols, box_indexs

def createSpanMatrix(rows, cols, box_indexs, bboxs):
    matrix = [[None for col in cols] for row in rows]
    for index, span in enumerate(box_indexs):
        matrix[span[0]][span[1]] = index
    
    # Duyệt qua từng phần tử của matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] is None: continue
            box_index = matrix[i][j]
            # lấy danh sách các bbox cùng hàng với box hiện tại và khác với nó
            same_row_bboxs = [box for box in bboxs if isSameRow(box, bboxs[box_index]) and box[3] - box[1] <= bboxs[box_index][3] - bboxs[box_index][1]]
            subrows = split_rows_columns([box[:2] for box in same_row_bboxs], 'row')
            # đếm số lượng x1 khác nhau trong same_row_bboxs
            rowspan = len(subrows)
            # lấy danh sách các bbox cùng cột với box hiện tại và khác với nó
            same_col_bboxs = [box for box in bboxs if isSameCol(box, bboxs[box_index]) and box[2] - box[0] <= bboxs[box_index][2] - bboxs[box_index][0]]
            subcols = split_rows_columns([box[:2] for box in same_col_bboxs], 'col')
            # đếm số lượng y1 khác nhau trong same_col_bboxs
            colspan = len(subcols)
            matrix[i][j] = (box_index, rowspan, colspan)
    return matrix

# Tạo cells từ bboxs, rows, cols, box_indexs, matrix
#cell = {'bbox': bbox, 'row_span': row_span, 'col_span': col_span, 'row':row_index, 'col':col_index}
def createCells(bboxs, box_indexs, matrix):
    cells = []
    for index, bbox in enumerate(bboxs):
        row_index, col_index = box_indexs[index]
        row_span, col_span = matrix[row_index][col_index][1:]
        cell = {'bbox': bbox, 'row_span': row_span, 'col_span': col_span, 'row':row_index, 'col':col_index}
        cells.append(cell)
    return cells

def createCell_img(cells, image):
    cells_imgs = []
    img_ok = image.copy()
    for cell in cells:
        x1,y1,x2,y2 = cell['bbox']
        # img_ok = cv2.cvtColor(image_removed, cv2.COLOR_BGR2GRAY)
        # img_ok = cv2.threshold(img_ok, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # chuyển lại sang ảnh màu
        # img_ok = cv2.cvtColor(img_ok[1], cv2.COLOR_GRAY2BGR)

        cropped_image = img_ok[int(y1):int(y2), int(x1):int(x2)]
        # resize về kích thước chiều nhỏ nhất là 50
        w,h = cropped_image.shape[:2]
        if w < h:
          scale_percent = 50 / cropped_image.shape[0]
        else:
          scale_percent = 50 / cropped_image.shape[1]
        
        width = int(cropped_image.shape[1] * scale_percent)
        height = int(cropped_image.shape[0] * scale_percent)
        cropped_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_AREA)

        # thêm 5px padding vào mỗi cạnh
        cropped_image = cv2.copyMakeBorder(cropped_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        cells_imgs += [(cropped_image, cell['bbox'])]
    return cells_imgs