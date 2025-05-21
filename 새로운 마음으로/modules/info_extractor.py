# modules/info_extractor.py

import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # 설치 경로 맞춰서 수정

def crop_region(img, region):
    """
    지정한 영역 자르기
    region: (left, top, right, bottom)
    """
    return img.crop(region)

def ocr_number(pil_img):
    """
    이미지에서 숫자 추출
    """
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(pil_img, config=config)
    text = text.strip().replace(" ", "").replace("\n", "")
    return int(text) if text.isdigit() else None

def extract_info(img):
    """
    전체 캡처 이미지에서 HP, 아이템 수, 타이머 추출
    ※ 좌표는 해상도 및 UI 위치에 맞게 조정 필요
    """
    # 예시 좌표 (왼쪽 상단 체력, 오른쪽 상단 적 체력, 중앙 하단 타이머, 하단 아이템)
    my_hp_region = (50, 50, 150, 90)
    enemy_hp_region = (1730, 50, 1830, 90)
    timer_region = (900, 960, 1020, 1000)
    item_region = (850, 1020, 1070, 1080)

    my_hp_img = crop_region(img, my_hp_region)
    enemy_hp_img = crop_region(img, enemy_hp_region)
    timer_img = crop_region(img, timer_region)
    item_img = crop_region(img, item_region)

    my_hp = ocr_number(my_hp_img)
    enemy_hp = ocr_number(enemy_hp_img)
    time_val = ocr_number(timer_img)
    item_count = ocr_number(item_img)

    return {
        "my_hp": my_hp,
        "enemy_hp": enemy_hp,
        "time": time_val,
        "item": item_count
    }
