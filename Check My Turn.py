import cv2
import numpy as np
import pyautogui
import time

template = cv2.imread("이미지 넣어", 0)#비교 이미지
w, h = template.shape[::-1]

def is_game_over():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8 # 유사도
    loc = np.where(res >= threshold)

    return len(list(zip(*loc[::-1]))) > 0
