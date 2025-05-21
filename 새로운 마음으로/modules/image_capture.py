# modules/image_capture.py

import pygetwindow as gw
import win32gui, win32ui, win32con
from PIL import Image
import time, os, threading
import numpy as np

# 설정
TITLE_KEYWORD = "Flash Player"
SAVE_DIR = "window_captures"
CAPTURE_INTERVAL = 0.03
DELETE_AFTER_SEC = 10

os.makedirs(SAVE_DIR, exist_ok=True)

def get_client_area(hwnd):
    left_top = win32gui.ClientToScreen(hwnd, (0, 0))
    right_bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:])
    return (*left_top, *right_bottom)

def auto_crop_white_margin(pil_img, threshold=250):
    img_np = np.array(pil_img)
    gray = np.mean(img_np, axis=2) if img_np.ndim == 3 else img_np
    h, w = gray.shape
    top, bottom = 0, h
    for y in range(h):
        if np.mean(gray[y]) < threshold:
            top = y
            break
    for y in range(h - 1, -1, -1):
        if np.mean(gray[y]) < threshold:
            bottom = y + 1
            break
    return pil_img.crop((0, top, w, bottom))

def capture_single_frame(idx):
    windows = gw.getWindowsWithTitle(TITLE_KEYWORD)
    if not windows:
        return None

    win = windows[0]
    hwnd = win._hWnd
    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return None

    left, top, right, bottom = get_client_area(hwnd)
    width, height = right - left, bottom - top

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

    cropped_img = auto_crop_white_margin(img)
    fname = os.path.join(SAVE_DIR, f"{idx:04d}.png")
    cropped_img.save(fname)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    return fname

def capture_window_loop():
    idx = 0
    while True:
        fname = capture_single_frame(idx)
        if fname:
            print(f"[Saved] {fname}")
            idx += 1
        time.sleep(CAPTURE_INTERVAL)

def delete_old_images_loop():
    while True:
        now = time.time()
        for fname in os.listdir(SAVE_DIR):
            if fname.endswith(".png"):
                fpath = os.path.join(SAVE_DIR, fname)
                if now - os.path.getmtime(fpath) > DELETE_AFTER_SEC:
                    try:
                        os.remove(fpath)
                        print(f"[Deleted] {fname}")
                    except Exception as e:
                        print(f"[Error] 삭제 실패: {fname} → {e}")
        time.sleep(1)

def start_capture_loop():
    t1 = threading.Thread(target=delete_old_images_loop, daemon=True)
    t2 = threading.Thread(target=capture_window_loop, daemon=True)
    t1.start()
    t2.start()
    print(f"[Start] 캡처 루프 시작됨 ('{TITLE_KEYWORD}' 대상)")

def get_latest_capture():
    files = sorted([
        f for f in os.listdir(SAVE_DIR) if f.endswith(".png")
    ], key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)

    if not files:
        return None

    latest_path = os.path.join(SAVE_DIR, files[0])
    return Image.open(latest_path)
