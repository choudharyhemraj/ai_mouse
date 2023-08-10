import pyautogui
import numpy as np

pyautogui.FAILSAFE = False
scr_width, scr_height = pyautogui.size()


def mouseClick():
    pyautogui.click()


def mouseScroll(amount):
    pyautogui.scroll(amount)


def moveCursor(landmarks, width, height, previous, action):
    thumb_tip = landmarks["Right"][4]
    index_finger_top = landmarks["Right"][8]
    thumb_x_cord = int(thumb_tip[0] * width)
    thumb_y_cord = int(thumb_tip[1] * height)
    index_x_cord = int(index_finger_top[0] * width)
    index_y_cord = int(index_finger_top[1] * height)
    x, y = (
        np.average([thumb_x_cord, index_x_cord]),
        np.average([thumb_y_cord, index_y_cord]),
    )
    curr_x, curr_y = pyautogui.position()
    new_x, new_y = curr_x, curr_y
    if previous["cord"] is not None:
        last_cord = previous["cord"]
        change_x = ((x - last_cord[0]) / width) * scr_width
        change_y = ((y - last_cord[1]) / height) * scr_height
        if abs(change_x) < 4 and abs(change_y) < 4:
            return
        new_x += change_x * 1.25
        new_y += change_y * 1.25
    previous["cord"] = (x, y)
    pyautogui.moveTo(new_x, new_y)
    if action == "drag":
        pyautogui.dragTo(new_x, new_y, button="left")
    else:
        pyautogui.moveTo(new_x, new_y)


def distance(x, y):
    return np.ceil(np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))


def detect_action(landmarks, width, height):
    left = None
    right = None
    if "Right" in landmarks:
        right = landmarks["Right"]
        right = list(map(lambda x: (int(x[0] * width), int(x[1] * height)), right))
    if "Left" in landmarks:
        left = landmarks["Left"]
        left = list(map(lambda x: (int(x[0] * width), int(x[1] * height)), left))
    if "Right" in landmarks:
        right_thumb_index_distance = distance(right[4], right[8])
        right_thumb_middle_distance = distance(right[4], right[12])

        indexs = [[8, 12], [7, 11], [6, 10], [5, 4], [4, 20], [4, 16], [16, 20]]
        scroll = True
        for index in indexs:
            d = distance(right[index[0]], right[index[1]])
            if d > 50:
                scroll = False

        if scroll:
            return "scroll"
        if right_thumb_index_distance < 32:
            return "move"
        if right_thumb_middle_distance < 32:
            return "click"
    if "Left" in landmarks:
        indexs = [[8, 12], [7, 11], [6, 10], [5, 4], [4, 20], [4, 16], [16, 20]]
        scroll_down = True
        for index in indexs:
            ld = distance(left[index[0]], left[index[1]])
            if ld > 50:
                scroll_down = False
        if scroll_down and left:
            return "scroll"
