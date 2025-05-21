# modules/action_executor.py

import pyautogui
import time

# 화면 해상도 기준값 (사용 환경에 맞게 수정)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# 마우스 클릭 시간 범위 (초)
CLICK_DURATION_MIN = 0.05
CLICK_DURATION_MAX = 0.5

class ActionExecutor:
    def __init__(self):
        pass

    def scale_action(self, action_vec):
        """
        행동 벡터를 실제 클릭 시간 및 마우스 좌표로 변환
        Args:
            action_vec: [-1, 1] 범위의 연속 벡터 [click_duration, x, y]

        Returns:
            (duration, x_pixel, y_pixel)
        """
        click_duration = CLICK_DURATION_MIN + (action_vec[0] + 1) / 2 * (CLICK_DURATION_MAX - CLICK_DURATION_MIN)
        x = int((action_vec[1] + 1) / 2 * SCREEN_WIDTH)
        y = int((action_vec[2] + 1) / 2 * SCREEN_HEIGHT)
        return click_duration, x, y

    def execute(self, action_vec):
        """
        행동을 실제로 수행 (마우스 이동 + 클릭)
        """
        duration, x, y = self.scale_action(action_vec)

        pyautogui.moveTo(x, y)
        pyautogui.mouseDown()
        time.sleep(duration)
        pyautogui.mouseUp()

        print(f"[ACTION] Clicked at ({x}, {y}) for {duration:.2f}s")
