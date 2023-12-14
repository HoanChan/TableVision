from utils.cv import cut_text_line
from PIL import Image
import cv2

def init_VietOCR():
    try:
        from vietocr.tool.predictor import Predictor
        print("Đã cài đặt thư viện nhận diện văn bản")
    except ModuleNotFoundError:
        # print("Đang cài đặt thư viện nhận diện văn bản...")
        # ! pip install --quiet vietocr
        print('Thư viện VietOCR chưa được cài đặt! Hãy cài bằng lệnh:\npip install --quiet vietocr')
    finally:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        config = Cfg.load_config_from_name('vgg_seq2seq')
        # config['weights'] = './weights/transformerocr.pth'
        # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False
        detector = Predictor(config)
        return detector
    

def detectText(image, detector):
    """
    Nhận diện văn bản trong ảnh.

    Tham số:
    - image: Ảnh cần nhận diện văn bản.

    Trả về:
    - texts: Danh sách các dòng văn bản đã nhận diện được.
    """
    lines = cut_text_line(image)
    if len(lines) == 0:
        return "", lines
    texts = []
    for line in lines:
        try:
            image = Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB))
            text = detector.predict(image)
            texts.append(text)
        except:
            texts.append('')
    texts = '\n'.join(texts)
    return texts , lines