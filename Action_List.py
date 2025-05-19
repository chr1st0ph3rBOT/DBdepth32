# 라이브러리
import pyautogui

# 기본 행동 정의 [클릭_in, 클릭_out, 커서_x, 커서_y]
class actions:
    def __init__(self, click_in=0.0, click_out=0.0, cursor_x=0.0, cursor_y=0.0):
        self.click_in = click_in
        self.click_out = click_out
        self.cursor_x = cursor_x
        self.cursor_y = cursor_y

    def initialization(self):
        self.click_in = 0.0
        self.click_out = 0.0
        self.cursor_x = 0.0
        self.cursor_y = 0.0

# 행동 실행 클래스
class executer:
    def __init__(self, action):
        self.action = action

    def execute(self):
        x = int(self.action.cursor_x * pyautogui.size()[0])
        y = int(self.action.cursor_y * pyautogui.size()[1])
        pyautogui.moveTo(x, y)

        if self.action.click_in > 0.5:
            pyautogui.mouseDown()
        if self.action.click_out > 0.5:
            pyautogui.mouseUp()