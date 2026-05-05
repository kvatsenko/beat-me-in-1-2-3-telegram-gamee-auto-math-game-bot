import time
import re
import threading
from pynput import keyboard

import mss
import numpy as np
import cv2
import Vision

import pyautogui

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = False

# Use 0 to capture all monitors, or 1/2/... to capture a specific monitor.
SCREEN_MONITOR_INDEX = 0

# Debug images are overwritten on each loop iteration.
DEBUG_FULL_IMAGE = "debug_full_detection.png"
DEBUG_OCR_IMAGE = "debug_capture.png"

# Extra padding around the detected equation before sending it to OCR.
EQUATION_PADDING_X = 40
EQUATION_PADDING_Y = 35

# Detection thresholds for the game area and answer buttons.
MIN_GAME_AREA = 80_000
MIN_BUTTON_AREA = 5_000

# Click settings.
CLICK_REPEAT = 1
CLICK_INTERVAL = 0.05


STOP_EVENT = threading.Event()


def on_press(key):
    try:
        if key == keyboard.Key.esc:
            print("Stop requested by Esc")
            STOP_EVENT.set()
            return False

        if hasattr(key, "char") and key.char and key.char.lower() in ("e", "у"):
            print("Stop requested by E")
            STOP_EVENT.set()
            return False
    except Exception as e:
        print("Keyboard listener error:", e)


def mss_to_pyautogui_point(x, y, monitor):
    """Convert MSS screen coordinates into pyautogui coordinates."""

    # With SCREEN_MONITOR_INDEX = 0, MSS usually returns global screen coordinates.
    py_x = int(x)
    py_y = int(y)

    # Keep the click point inside the visible screen bounds.
    screen_w, screen_h = pyautogui.size()
    py_x = max(0, min(screen_w - 1, py_x))
    py_y = max(0, min(screen_h - 1, py_y))

    return py_x, py_y




def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Reduce small visual noise before thresholding.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Keep bright text and suppress the colored background.
    _, th = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    # Upscale the cropped equation to improve OCR accuracy.
    th = cv2.resize(th, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return th


def find_equation_roi(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # Ignore tiny noise and very large UI elements.
        if area < 80 or area > 50000:
            continue

        # Equation characters are usually tall and bright.
        if h < 18 or w < 5:
            continue

        boxes.append((x, y, w, h))

    if not boxes:
        return None

    # Group bright objects that are horizontally aligned.
    boxes.sort(key=lambda b: b[1])
    groups = []

    for box in boxes:
        x, y, w, h = box
        placed = False
        for group in groups:
            gy_values = [b[1] + b[3] / 2 for b in group]
            group_center_y = sum(gy_values) / len(gy_values)
            if abs((y + h / 2) - group_center_y) < 45:
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])

    best_group = None
    best_score = 0

    for group in groups:
        if len(group) < 2:
            continue

        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[0] + b[2] for b in group)
        y2 = max(b[1] + b[3] for b in group)
        w = x2 - x1
        h = y2 - y1

        if w < 40 or h < 25:
            continue

        score = len(group) * 100 + w + h
        if score > best_score:
            best_score = score
            best_group = (x1, y1, x2, y2)

    if best_group is None:
        return None

    x1, y1, x2, y2 = best_group
    x1 = max(0, x1 - EQUATION_PADDING_X)
    y1 = max(0, y1 - EQUATION_PADDING_Y)
    x2 = min(img_bgr.shape[1], x2 + EQUATION_PADDING_X)
    y2 = min(img_bgr.shape[0], y2 + EQUATION_PADDING_Y)

    return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}


def find_game_elements(img_bgr):
    """Detect the game panel, equation area, and three answer buttons."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Detect the cyan game background.
    cyan_mask = cv2.inRange(hsv, (80, 50, 50), (100, 255, 255))
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    game_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(game_contour) < MIN_GAME_AREA:
        return None

    gx, gy, gw, gh = cv2.boundingRect(game_contour)
    game_roi = {"left": gx, "top": gy, "width": gw, "height": gh}

    game_img = img_bgr[gy:gy + gh, gx:gx + gw]
    game_hsv = hsv[gy:gy + gh, gx:gx + gw]

    # Detect the purple answer buttons inside the game area.
    purple_mask = cv2.inRange(game_hsv, (125, 25, 25), (170, 255, 190))
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    button_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buttons = []

    for c in button_contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < MIN_BUTTON_AREA:
            continue
        if w < gw * 0.35:
            continue
        if h < 25:
            continue

        buttons.append((gx + x, gy + y, w, h))

    buttons = sorted(buttons, key=lambda b: b[1])[:3]
    if len(buttons) != 3:
        return None

    button_centers = {
        index + 1: (x + w // 2, y + h // 2)
        for index, (x, y, w, h) in enumerate(buttons)
    }

    # Search for the equation only above the first answer button.
    first_button_y = buttons[0][1]
    eq_top_local = max(0, int(gh * 0.30))
    eq_bottom_local = max(eq_top_local + 50, first_button_y - gy)
    equation_area = game_img[eq_top_local:eq_bottom_local, :]

    equation_roi_local = find_equation_roi(equation_area)
    if equation_roi_local is None:
        return None

    equation_roi = {
        "left": gx + equation_roi_local["left"],
        "top": gy + eq_top_local + equation_roi_local["top"],
        "width": equation_roi_local["width"],
        "height": equation_roi_local["height"],
    }

    return {
        "game_roi": game_roi,
        "equation_roi": equation_roi,
        "button_centers": button_centers,
    }


def save_detection_debug(full_bgr, elements):
    debug = full_bgr.copy()

    game = elements["game_roi"]
    gx1 = game["left"]
    gy1 = game["top"]
    gx2 = gx1 + game["width"]
    gy2 = gy1 + game["height"]
    cv2.rectangle(debug, (gx1, gy1), (gx2, gy2), (255, 0, 0), 3)

    roi = elements["equation_roi"]
    x1 = roi["left"]
    y1 = roi["top"]
    x2 = x1 + roi["width"]
    y2 = y1 + roi["height"]
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)

    for answer, (x, y) in elements["button_centers"].items():
        cv2.circle(debug, (x, y), 14, (0, 0, 255), -1)
        cv2.putText(
            debug,
            str(answer),
            (x - 8, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(DEBUG_FULL_IMAGE, debug)


def mac_os_ocr(cv_img):
    ok, buffer = cv2.imencode(".png", cv_img)
    if not ok:
        return ""

    data = buffer.tobytes()
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()

    success, error = handler.performRequests_error_([request], None)
    if not success:
        print("OCR error:", error)
        return ""

    results = request.results()
    if not results:
        return ""

    texts = []
    for r in results:
        candidate = r.topCandidates_(1)
        if candidate and len(candidate) > 0:
            texts.append(str(candidate[0].string()))

    return " ".join(texts).strip()


def normalize_ocr_text(text):
    # Common OCR mistakes for this game font.
    replacements = {
        "—": "-",
        "–": "-",
        "−": "-",
        "×": "+",
        "x": "+",
        "X": "+",
        "÷": "-",
        "=": "",
        "?": "",
        " ": "",
        "O": "0",
        "o": "0",
        "l": "1",
        "I": "1",
        "|": "1",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def looks_like_math(text):
    return bool(re.search(r"\d", text)) and bool(re.search(r"[+\-]", text))


def solve(text):
    normalized = normalize_ocr_text(text)
    clean = re.sub(r"[^0-9+\-]", "", normalized)

    print("RAW:   ", repr(text))
    print("NORM:  ", repr(normalized))
    print("CLEAN: ", repr(clean))

    if len(clean) < 3:
        return None

    try:
        result = eval(clean, {"__builtins__": None}, {})
        print("RESULT:", result)
        return result
    except Exception as e:
        print("EVAL ERROR:", e)
        return None


def click_answer(answer, elements, monitor):
    point = elements["button_centers"].get(answer)
    if point is None:
        print(f"No button coordinates found for answer {answer}")
        return

    mss_x, mss_y = point
    py_x, py_y = mss_to_pyautogui_point(mss_x, mss_y, monitor)

    print(f"Clicking answer {answer}")
    print(f"  button mss=({mss_x}, {mss_y}) -> pyautogui=({py_x}, {py_y})")
    print(f"  mouse before={pyautogui.position()}")
    print(f"  final click point=({py_x}, {py_y})")

    pyautogui.moveTo(py_x, py_y, duration=0.15)
    time.sleep(0.03)

    for _ in range(CLICK_REPEAT):
        pyautogui.mouseDown()
        time.sleep(0.03)
        pyautogui.mouseUp()
        time.sleep(CLICK_INTERVAL)

    print(f"  mouse after={pyautogui.position()}")


def main():
    print("OCR bot started. Press E, Esc, or Ctrl+C to stop.")
    last_text = ""

    STOP_EVENT.clear()
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    with mss.mss() as sct:
        monitor = sct.monitors[SCREEN_MONITOR_INDEX]
        print("MSS monitor:", monitor)
        print("pyautogui screen:", pyautogui.size())

        while not STOP_EVENT.is_set():
            full_img = np.array(sct.grab(monitor))
            full_bgr = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

            elements = find_game_elements(full_bgr)
            if elements is None:
                print("Game screen, buttons, or equation not found")
                STOP_EVENT.wait(0.2)
                continue

            roi = elements["equation_roi"]
            img_bgr = full_bgr[
                roi["top"]:roi["top"] + roi["height"],
                roi["left"]:roi["left"] + roi["width"],
            ]

            save_detection_debug(full_bgr, elements)

            proc = preprocess(img_bgr)

            # Save the OCR input crop for troubleshooting.
            cv2.imwrite(DEBUG_OCR_IMAGE, proc)

            raw_text = mac_os_ocr(proc)

            if raw_text:
                if raw_text != last_text:
                    last_text = raw_text
                    print("OCR:", repr(raw_text))

                if looks_like_math(raw_text):
                    res = solve(raw_text)
                    if res in [1, 2, 3]:
                        print(f"Equation: {raw_text} | Answer: {res}")
                        click_answer(res, elements, monitor)
                        STOP_EVENT.wait(0.5)  # Delay before the next click.
                    else:
                        print("Equation found, but the answer is not 1, 2, or 3")
                else:
                    print("Skipping non-math OCR text:", repr(raw_text))

                print("-" * 40)

            STOP_EVENT.wait(0.08)

    listener.stop()
    print("Script stopped")


if __name__ == "__main__":
    main()