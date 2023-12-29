import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.bbox import is_bbox_inside

def display_images_with_labels(image_labels, layout, size=(10, 10), show_axis=True):
    """
    Hiển thị các ảnh cùng với nhãn bằng matplotlib.pyplot.

    Tham số:
    - image_labels: Danh sách các tuple (ảnh, nhãn) cần hiển thị.
    - layout: Tuple (rows, columns) chỉ định cách bố trí ảnh trên lưới.
    - size: Tuple (width, height) chỉ định kích thước của mỗi ảnh.
    - show_axis: Bool, xác định xem có hiển thị trục hay không.
    """
    rows, columns = layout
    total_images = len(image_labels)

    # Tính số lượng ảnh cần thêm vào để điền đầy lưới
    num_padding = rows * columns - total_images

    # Thêm ảnh trống vào danh sách nếu cần
    image_labels += [(None, None)] * num_padding

    # Tạo subplot với tỉ lệ cố định
    fig, axes = plt.subplots(rows, columns, figsize=(size[0]*columns, size[1]*rows), subplot_kw={'aspect': 'equal'})

    # Hiển thị ảnh cùng với nhãn trên các ô subplot
    for i, (image, label) in enumerate(image_labels):
        ax = axes.flat[i]
        if image is not None:
            # Chuyển đổi màu từ BGR sang RGB để hiển thị đúng
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
        if label is not None:
            ax.set_title(label)
        if not show_axis:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def preProcessing(image, minRecSize = 5000):    
    """
    Tiền xử lý ảnh.
    
    Tham số:
    - image: Ảnh cần tiền xử lý.
    - minRecSize: Kích thước tối thiểu của vùng hình chữ nhật để được xem là vùng hình chữ nhật màu đen.
    
    Kết quả:
    - data: Danh sách các tuple (ảnh, nhãn) đã tiền xử lý.
    - result: Ảnh đã tiền xử lý.
    """
    data = []
    # # resize về chiều nhỏ nhất là 1000
    # w,h = image.shape[:2]
    # if w < h:
    #     scale_percent = 2000 / image.shape[0]
    # else:
    #     scale_percent = 2000 / image.shape[1]
    # width = int(image.shape[1] * scale_percent)
    # height = int(image.shape[0] * scale_percent)
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    data+=[(image,'original')]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data+=[(gray,'gray')]
    # Áp dụng ngưỡng
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    data+=[(binary,'binary')]
    # tìm các vùng hình chữ nhật màu đen
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #lọc các vùng hình chữ nhật theo kích thước 
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > minRecSize ]
    bboxs = [cv2.boundingRect(cnt) for cnt in contours]
    # sắp xếp các bbox theo diện tích tăng dần
    bboxs.sort(key=lambda box: box[2]*box[3])
    # chuyển sang x1,y1,x2,y2
    bboxs = [(x,y,x+w,y+h) for x,y,w,h in bboxs]
    # bỏ các bbox chứa bbox khác
    for i,box in enumerate(bboxs):
        for j in range(i+1,len(bboxs)):
            if is_bbox_inside(bboxs[j],box):
                bboxs.pop(j)
                break
    # đảo ngược màu sắc của các vùng hình chữ nhật
    newbinary = binary.copy()
    for box in bboxs:
        x1,y1,x2,y2 = box
        # print(cnt)
        # lấy ra vùng hình chữ nhật trong binary
        roi = binary[y1:y2, x1:x2]
        # kiểm tra màu sắc chủ đạo của vùng hình chữ nhật
        if np.mean(roi) > 128:
            continue
        # đảo ngược màu sắc
        roi = cv2.bitwise_not(roi)
        # gán lại vùng hình chữ nhật đã đảo ngược màu sắc vào binary
        newbinary[y1:y2, x1:x2] = roi
    # chuyển binary sang ảnh màu
    result = cv2.cvtColor(newbinary, cv2.COLOR_GRAY2BGR)
    # vẽ các bbox lên ảnh
    # for box in bboxs:
    #     x1, y1, x2, y2 = box
    #     cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    data+=[(result,'image')]
    return data, result

def deskew_image(image):
    """
    Chỉnh sửa góc nghiêng của ảnh.

    Tham số:
    - image: Ảnh cần chỉnh sửa góc nghiêng.

    Kết quả:
    - deskewed_image: Ảnh đã chỉnh sửa góc nghiêng.
    - angle: Góc nghiêng của ảnh.
    """
    # Làm mờ ảnh để loại bỏ nhiễu
    blur_image = cv2.medianBlur(image, 5)

    # Chuyển đổi sang ảnh grayscale và đảo ngược màu
    grayscale_image = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Xác định ngưỡng 
    _, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Tìm tọa độ các điểm ảnh khác 0 (điểm ảnh màu trắng)
    coordinates = np.column_stack(np.where(threshold_image > 0))

    # Xác định góc xoay của ảnh
    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Xoay ảnh để chỉnh sửa góc nghiêng và điều chỉnh kích thước ảnh để xoay không bị cắt ảnh
    height, width = blur_image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed_image, angle

def rotate_image(image, angle):
    # Xác định kích thước ảnh
    height, width = image.shape[:2]

    # Tính toán ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

    # Tính toán kích thước mới của ảnh sau khi xoay
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Cập nhật ma trận xoay để điều chỉnh kích thước ảnh
    rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
    rotation_matrix[1, 2] += (new_height / 2) - (height / 2)

    # Thực hiện xoay và điều chỉnh kích thước ảnh
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated_image

def trim_white(image):
    # Chuyển đổi ảnh sang không gian màu xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng ngưỡng nhị phân để tạo mask chỉ chứa các điểm ảnh trắng
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Tìm các chỉ số hàng và cột không chứa giá trị 0 (màu trắng) đầu tiên
    rows = np.where(np.any(thresh != 0, axis=1))[0]
    cols = np.where(np.any(thresh != 0, axis=0))[0]

    # Tính toán các giới hạn của bounding box
    y1, y2 = rows[0], rows[-1] + 1
    x1, x2 = cols[0], cols[-1] + 1

    # Cắt và trả về phần ảnh đã được trim
    trimmed_image = image[y1:y2, x1:x2]

    # Thêm 20 pixel vào mỗi cạnh để tránh việc cắt bớt các ký tự
    trimmed_image = cv2.copyMakeBorder(trimmed_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    #resize về chiều nhỏ nhất là 1000
    w,h = trimmed_image.shape[:2]
    if w < h:
      scale_percent = 2000 / trimmed_image.shape[0]
    else:
      scale_percent = 2000 / trimmed_image.shape[1]

    width = int(trimmed_image.shape[1] * scale_percent)
    height = int(trimmed_image.shape[0] * scale_percent)
    trimmed_image = cv2.resize(trimmed_image, (width, height), interpolation=cv2.INTER_AREA)

    return trimmed_image

def remove_regions(image, mask):
    """
    Loại bỏ các vùng trong ảnh sử dụng mặt nạ.

    Tham số:
    - image: Ảnh gốc.
    - mask: Mặt nạ có kích thước giống với ảnh, với giá trị 0 ở các vùng cần loại bỏ và 1 ở các vùng khác.

    Trả về:
    - Ảnh đã loại bỏ các vùng cần loại bỏ.
    """
    masked_image = image.copy()
    masked_image[mask != 0] = [255, 255, 255]
    return masked_image

def draw_text_in_center(img, text, box, size = 0.7, color = (0, 0, 255)):
    x1, y1, x2, y2 = box

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    text_width = text_size[0]
    text_height = text_size[1]
    text_origin = (center[0] - text_width // 2, center[1] + text_height // 2)

    cv2.putText(img, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

def draw_bboxs(image, bboxs, texts = lambda x: str(x)):    
    img_width = image.shape[1]
    # Sắp xếp các bboxs theo vị trí từ trên xuống dưới, từ trái qua phải
    bboxs = sorted(bboxs, key=lambda box: box[1] * img_width + box[0])    
    img_bboxs = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # vẽ bboxs
    for index, box in enumerate(bboxs, start=1):
        cv2.rectangle(img_bboxs, box[:2], box[2:], (0, 255, 0), 3)
        draw_text_in_center(img_bboxs, texts(index), box)
    return img_bboxs


def cut_text_line(image):
    """
    Cắt các dòng văn bản từ ảnh.

    Tham số:
    - image: Ảnh cần xử lý.

    Trả về:
    - images: Danh sách các dòng văn bản đã cắt được.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilated = cv2.dilate(bw, kernel, iterations=10) # mở rộng để lấp đầy hàng
    # return [dilated]
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    images = []
    # img_cnt = image.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > 100 and w > h and h > 5:
            # cv2.drawContours(img_cnt, contours, -1, (0, w%255, 0), 2)
            cropped = image[y-3:y+h+3, x:]
            images+=[cropped]
    # return [img_cnt]
    if len(images) == 0:
        return []
    # Tìm độ cao lớn nhất của các ảnh
    max_height = max([img.shape[0] for img in images])
    # xoá các image có độ cao nhỏ hơn nhiều so với max_height
    images = [img for img in images if img.shape[0] > max_height / 2]
    return images[::-1]

# vẽ các trọng tâm của bbox lên img mới
def drawCenters(img, centers, size = 20, color = (0, 0, 255), thickness = 3):
    image = np.zeros_like(img)
    for center in centers:        
        cv2.line(image, (center[0]-size//2, center[1]-size//2), (center[0]+size//2, center[1]+size//2), color, thickness)
        cv2.line(image, (center[0]+size//2, center[1]-size//2), (center[0]-size//2, center[1]+size//2), color, thickness)
        # cv2.circle(img, center, 1, (255, 255, 255), -1)
    return image

# Vẽ các hàng và cột lên ảnh dựa vào điểm đầu và điểm cuối của mỗi hàng/cột
def draw_rows_columns(img, rows, columns):
    img = img.copy()
    for row in rows:
        cv2.line(img, row[0], row[-1], (0, 255, 0), 3)
    for column in columns:
        cv2.line(img, column[0], column[-1], (255, 0, 255), 3)
    return img