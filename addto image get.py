template = cv2.imread("이미지 경로", 0)
def is_game_over(pil_img):
    img_gray = pil_img.convert("L")
    img_np = np.array(img_gray)

    res = cv2.matchTemplate(img_np, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    return len(list(zip(*loc[::-1]))) > 0
