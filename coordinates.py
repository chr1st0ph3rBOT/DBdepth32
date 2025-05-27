import cv2

screen_path = r"C:\Mingdda\1.png"
template_path = r"C:\Mingdda\2.png"


screen = cv2.imread(screen_path)
template = cv2.imread(template_path)

if template.shape[0] > screen.shape[0] or template.shape[1] > screen.shape[1]:
    raise ValueError("❌ 템플릿 이미지가 전체 화면보다 큽니다. 다시 자르세요.")

result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFFNORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)

x, y = max_loc
print(f"🎯 wind 게이지 좌표 (좌상단): ({x}, {y})")