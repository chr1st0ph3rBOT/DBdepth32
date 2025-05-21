# modules/preprocessing.py

from PIL import ImageOps
import numpy as np

def preprocess(pil_img, size=(84, 84)):
    """
    이미지 전처리: 흑백 변환 + 크기 조정 + 정규화 + 차원 추가

    Args:
        pil_img (PIL.Image): 원본 이미지
        size (tuple): 출력 크기, 기본값 (84, 84)

    Returns:
        np.ndarray: 전처리된 이미지 (shape: [84, 84, 1])
    """
    # 흑백 변환
    gray = ImageOps.grayscale(pil_img)

    # 크기 조정
    resized = gray.resize(size)

    # numpy 배열 변환 및 정규화 (0~1)
    arr = np.array(resized).astype(np.float32) / 255.0

    # 차원 추가: (H, W) → (H, W, 1)
    arr = np.expand_dims(arr, axis=-1)

    return arr
