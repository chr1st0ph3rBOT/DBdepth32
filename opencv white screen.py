import cv2
import numpy as np
import pyautogui
import time

# '왕따 WINS!' 텍스트 이미지 템플릿
template = cv2.imread("C:/Users/heung/Downloads/game_end_image(cut1).png", 0)
w, h = template.shape[::-1]

def is_game_over():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)

    return len(list(zip(*loc[::-1]))) > 0   

# 사용 예
while True:
    if is_game_over():
        print("게임 종료 화면입니다!")
