import pygetwindow as gw
import win32gui, win32ui, win32con
from PIL import Image
import time
import os
import threading
import numpy as np

# 설정
TITLE_KEYWORD = "Flash Player"     # 캡처할 창 제목 일부
SAVE_DIR = "window_captures"
CAPTURE_INTERVAL = 0.03
DELETE_AFTER_SEC = 10

os.makedirs(SAVE_DIR, exist_ok=True)

# 클라이언트 영역 계산 함수
def get_client_area(hwnd):
    left_top = win32gui.ClientToScreen(hwnd, (0, 0))
    right_bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:])
    left, top = left_top
    right, bottom = right_bottom
    return left, top, right, bottom

# 자동 흰색 여백 감지 후 crop
def auto_crop_white_margin(pil_img, threshold=250):
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np

    h, w = gray.shape
    top, bottom = 0, h

    # 상단 여백 감지
    for y in range(h):
        if np.mean(gray[y]) < threshold:
            top = y
            break

    # 하단 여백 감지
    for y in range(h - 1, -1, -1):
        if np.mean(gray[y]) < threshold:
            bottom = y + 1
            break

    # Crop
    cropped = pil_img.crop((0, top, w, bottom))
    return cropped

# 창 캡처 루프
def capture_window_loop():
    idx = 0
    while True:
        windows = gw.getWindowsWithTitle(TITLE_KEYWORD)
        if not windows:
            print(f"[ERROR] '{TITLE_KEYWORD}' 창을 찾을 수 없습니다.")
            time.sleep(CAPTURE_INTERVAL)
            continue

        win = windows[0]
        hwnd = win._hWnd

        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            print(f"[SKIP] 창이 비활성화되었거나 최소화됨")
            time.sleep(CAPTURE_INTERVAL)
            continue

        left, top, right, bottom = get_client_area(hwnd)
        width = right - left
        height = bottom - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (width, height),
                      mfcDC,
                      (left - win32gui.GetWindowRect(hwnd)[0], top - win32gui.GetWindowRect(hwnd)[1]),
                      win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
                               bmpstr, "raw", "BGRX", 0, 1)

        # 🔥 자동 여백 제거
        cropped_img = auto_crop_white_margin(img)

        fname = os.path.join(SAVE_DIR, f"{idx:04d}.png")
        cropped_img.save(fname)
        print(f"[Saved] {fname}")

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        idx += 1
        time.sleep(CAPTURE_INTERVAL)

# 오래된 이미지 삭제 루프
def delete_old_images_loop():
    while True:
        now = time.time()
        for fname in os.listdir(SAVE_DIR):
            if fname.endswith(".png"):
                fpath = os.path.join(SAVE_DIR, fname)
                try:
                    if now - os.path.getmtime(fpath) > DELETE_AFTER_SEC:
                        os.remove(fpath)
                        print(f"[Deleted] {fname}")
                except Exception as e:
                    print(f"[Error] 삭제 실패: {fname} → {e}")
        time.sleep(1)

# 실행 시작
if __name__ == "__main__":
    print(f"🔥 '{TITLE_KEYWORD}' 창만 캡처 + 여백 제거 + 10초 후 자동 삭제 중! (종료: Ctrl+C)")

    t = threading.Thread(target=delete_old_images_loop, daemon=True)
    t.start()

    try:
        capture_window_loop()
    except KeyboardInterrupt:
        print("\n🛑 캡처 중단됨.")
