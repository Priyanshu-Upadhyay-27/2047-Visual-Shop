import cv2
import autopy as ap
import mediapipe as mp
import numpy as np
import time
import math
import pyautogui as pyag

wCam = 640    # Setting dimensions for camera screen
hCam = 480

frameR = 100     # frame reduction

smoothening = 3
plocX, plocY = 0, 0
clocX, clocY = 0, 0

mpHands = mp.solutions.hands  # hand class
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)     # Setting of object to access the camera and read frames
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

wScr, hScr = ap.screen.size()    # my window screen dimensions

left_click_flag = False
right_click_flag = False
click_display_duration = 2  # Frames for which the click indicator should be displayed
left_click_timer = 0
right_click_timer = 0

while True:
    ret, pad = cap.read()
    pad = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
    results = hands.process(pad)
    Lmlist = []
    Xlist = []
    Ylist = []
    # box around my palm
    bbox = []
    # box in which movement is allowed
    cv2.rectangle(pad, (frameR, frameR), (wCam - frameR, hCam - frameR), (128, 128, 128), 5)
    cv2.line(pad, (320, 330), (320, 380), (128, 128, 128), 5, cv2.LINE_8)
    cv2.putText(pad, "Priyanshu Upadhyay", (110, 120), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                0.5, (255, 0, 255), 1)
    cv2.putText(pad, "Your Hand is the New Cursor", (230, 310), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                0.7, (255, 0, 255), 1)
    cv2.line(pad, (100, 330), (540, 330), (128, 128, 128), 5, cv2.LINE_8)
    if not left_click_flag:
        cv2.rectangle(pad, (101, 330), (318, 378), (50, 50, 50), -5)  # left button
    if not right_click_flag:
        cv2.rectangle(pad, (321, 330), (538, 378), (50, 50, 50), -5)  # right button
    # Getting coordinates of each landmark with their id and drawing a bounding box around palm
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, d = pad.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                Xlist.append(cx)
                Ylist.append(cy)
                Lmlist.append([id, cx, cy])
            Xmin, Xmax = min(Xlist), max(Xlist)
            Ymin, Ymax = min(Ylist), max(Ylist)
            bbox = Xmin, Ymin, Xmax, Ymax
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            #print(bbox_area)
            #print(Lmlist)
            if True:
                cv2.rectangle(pad, (Xmin, Ymin), (Xmax, Ymax), (0, 255, 0), 2)
            mpDraw.draw_landmarks(pad, handLms, mpHands.HAND_CONNECTIONS)

        # When hand is detected, checking whether index and middle finger is open or close and act accordingly.
        if len(Lmlist) != 0:
            x1, y1 = Lmlist[8][1], Lmlist[8][2]
            x2, y2 = Lmlist[12][1], Lmlist[12][2]
            x4, y4 = Lmlist[4][1], Lmlist[4][2]
            x5, y5 = Lmlist[16][1], Lmlist[16][2]
            #print(x1, y1, x2, y2)
            ind_mid = math.hypot(x2 - x1, y2 - y1)
            th_ind = math.hypot(x4 - x1, y4 - y1)
            th_mid = math.hypot(x4 - x2, y4 - y2)

            # When index is open : Moving
            if Lmlist[8][2] < Lmlist[6][2]:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                ap.mouse.move(wScr - clocX, clocY)
                cv2.circle(pad, (x1, y1), 5, (255, 255, 0), cv2.FILLED)
                plocX, plocY = clocX, clocY
                cv2.putText(pad, "Cursor Moving", (85, 90), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 255), 2)
            #Left-Click
            if Lmlist[4][1] < Lmlist[3][1] and th_mid > 30:
                ap.mouse.click(ap.mouse.Button.LEFT)
                #time.sleep(0.1)

                left_click_flag = True
                left_click_timer = click_display_duration

                cv2.putText(pad, " + Left Click", (325, 90), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 255), 2)
            if th_mid < 30:
                ap.mouse.click(ap.mouse.Button.RIGHT)

                right_click_flag = True
                right_click_timer = click_display_duration

                cv2.putText(pad, " + Right Click", (325, 90), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 255), 2)
                #time.sleep(1)
            #Scrolling
            if Lmlist[12][2] < Lmlist[10][2] and Lmlist[8][2] < Lmlist[6][2] and Lmlist[16][2] > Lmlist[14][2]:
                if ind_mid > 40:
                    pyag.scroll(10)  # Scrolls down by 5 units
                    cv2.putText(pad, " + Scrolling Up", (325, 90), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 255), 2)
                    #time.sleep(0.1)
                if ind_mid < 40:
                    pyag.scroll(-10)  # Scrolls up by 5 units
                    cv2.putText(pad, " + Scrolling Down", (325, 90), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 255), 2)
                    #time.sleep(0.1)
            # Debug prints
            print(f"(thumb tip)X: {Lmlist[4][1]}, (thumb IP joint)X: {Lmlist[3][1]}")
            print(f"(pinky)Y: {Lmlist[20][2]}, (pinky IP joint)Y: {Lmlist[18][2]}")
            print(f"(middle)Y: {Lmlist[12][2]}, (middle IP joint)Y: {Lmlist[10][2]}")
            print(f"th_mid: {th_mid}")
    else:
        cv2.putText(pad, "There must be a single hand", (65, 440), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 0, 0), 2)
    # Update the timers and flags for click display
    if left_click_timer > 0:
        left_click_timer -= 1
    else:
        left_click_flag = False

    if right_click_timer > 0:
        right_click_timer -= 1
    else:
        right_click_flag = False
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(pad, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    pad = cv2.cvtColor(pad, cv2.COLOR_RGB2BGR)
    cv2.imshow("Mouse Pad", pad)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
