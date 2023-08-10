import cv2
from mediapipe import ImageFormat
import mediapipe as mp
from HandLandmark import draw_landmarks_on_image, detector
import numpy as np
import pyautogui

from utils import moveCursor, detect_action, mouseClick, mouseScroll


def mouse():
    video = cv2.VideoCapture(0)
    previousMove = {"cord": None}
    previousDrag = {"cord": None}
    doubleClicked = False
    while video.isOpened():
        success, frame = video.read()
        height, width, channels = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image = mp.Image(image_format=ImageFormat.SRGB, data=image)
        detection_result = detector.detect(image)

        annotated_image, landmarks = draw_landmarks_on_image(
            image.numpy_view(), detection_result, drawLandmarks=True
        )
        # print(landmarks)
        if "Right" in landmarks:
            action = detect_action(landmarks, width, height)
            print(action)
            if action == "move":
                moveCursor(landmarks, width, height, previousMove, action)
            else:
                previousMove = {"cord": None}
            if action == "click":
                mouseClick()
            if action == "scroll":
                mouseScroll(-1)
        if "Left" in landmarks:
            action = detect_action(landmarks, width, height)
            if action == "scroll":
                mouseScroll(1)
        cv2.imshow("frame", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mouse()
