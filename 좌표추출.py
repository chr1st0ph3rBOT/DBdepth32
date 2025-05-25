import cv2

# 이미지 경로
screen_path = r"C:\Mingdda\1.png"
template_path = r"C:\Mingdda\2.png"


# 이미지 불러오기
screen = cv2.imread(screen_path)
template = cv2.imread(template_path)

# 크기 확인 (오류 방지용)
if template.shape[0] > screen.shape[0] or template.shape[1] > screen.shape[1]:
    raise ValueError("❌ 템플릿 이미지가 전체 화면보다 큽니다. 다시 자르세요.")

# 템플릿 매칭
result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)

# 좌표 출력
x, y = max_loc
print(f"🎯 wind 게이지 좌표 (좌상단): ({x}, {y})")
