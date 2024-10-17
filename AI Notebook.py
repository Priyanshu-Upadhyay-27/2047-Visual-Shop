import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import math


# My DL model
genai.configure(api_key="AIzaSyC2YBSBArq1UxYk456q6FQ9mDoA7OoYrKc")
model = genai.GenerativeModel("gemini-1.5-flash")

#Function for sending data to the model.
def sendToAI(model, ImgCanvas , fingers):
    #print(thumb_ring)
    if (thumb_ring < 30 and Lmlist[20][2] < Lmlist[18][2]) and not Lmlist[20][2] > Lmlist[18][2]:
        pil_image = Image.fromarray(ImgCanvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        print(response.text)
        return response.text

# Some important initialization
penThickness = 15

# displaying the canvas at the header
folderPath = "Header"
mylist = os.listdir(folderPath)
overLayList = []
for imPath in mylist:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overLayList.append(image)

header = overLayList[0]
drawColor = (250, 250, 0)  # Default color

# Initializing Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8
)
mpDraw = mp.solutions.drawing_utils

#Initializing Open CV
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#Some important initialization
xp, yp = 0, 0
ImgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    success, board = cap.read()
    board = cv2.flip(board, 1)
    boardRGB = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    results = hands.process(boardRGB) # processed result
    Lmlist = [] #initializing empty list, which will contain coordinates of landmarks

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, d = board.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                Lmlist.append([id, cx, cy])

                mpDraw.draw_landmarks(board, handLms, mpHands.HAND_CONNECTIONS) # drawing markers
        #checking for left or right hand
        for handedness in results.multi_handedness:
            hand_label = handedness.classification[0].label

        # providing commands after detection
        if len(Lmlist) != 0:
            x1, y1 = Lmlist[8][1], Lmlist[8][2]
            x2, y2 = Lmlist[12][1], Lmlist[12][2]
            x3, y3 = Lmlist[4][1], Lmlist[4][2]
            x4, y4 = Lmlist[16][1], Lmlist[16][2]
            thumb_ring = math.hypot(x4 - x3, y4 - y3)
            fingers = [] # initializing an empty list

            # appending 1 to fingers[] because we will be denoting 1 as open finger and vice versa for 0.
            if hand_label == "Left" and Lmlist[4][1] > Lmlist[2][1]:
                fingers.append(1)
            elif hand_label == "Right" and Lmlist[4][1] < Lmlist[2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in [8, 12, 16, 20]:
                if Lmlist[id][2] < Lmlist[id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            sendToAI(model, ImgCanvas, fingers) # Model is called

            # Commands are provided according to the fingers position
            # Selection Mode
            if fingers[1] and fingers[2] and not (fingers[3] and fingers[4]):
                #setting up of making header dynamic using different canvas
                if y1 < 125:
                    if 0 < x1 < 140:
                        header = overLayList[0]
                        drawColor = (250, 0, 0)  # Blue
                        penThickness = 15
                    elif 265 < x1 < 390:
                        header = overLayList[1]
                        drawColor = (0, 255, 255)  # Yellow
                        penThickness = 15
                    elif 545 < x1 < 660:
                        header = overLayList[2]
                        drawColor = (0, 255, 0)  # Green
                        penThickness = 15
                    elif 850 < x1 < 975:
                        header = overLayList[3]
                        drawColor = (0, 0, 255)  # Red
                        penThickness = 15
                    elif 1080 < x1 < 1265:
                        header = overLayList[4]
                        drawColor = (0, 0, 0)  # Black
                        penThickness = 50

                cv2.rectangle(board, (x1, y1 - 25), (x2, y2 + 25), drawColor, -1)#finger pointer
                xp, yp = x1, y1 # re - arrangement
            # Writting Mode
            elif fingers[1] and not (fingers[2] and fingers[3] and fingers[4]):
                cv2.circle(board, (x1, y1), 10, drawColor, -1)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(board, (xp, yp), (x1, y1), drawColor, penThickness)
                cv2.line(ImgCanvas, (xp, yp), (x1, y1), drawColor, penThickness)
                xp, yp = x1, y1
            elif fingers == [1,1,1,1,0]:
                x = sendToAI(model, ImgCanvas, fingers)
                cv2.putText(board, x, (900, 400), cv2.FONT_HERSHEY_PLAIN, 20, drawColor, 2)
            # clearing all
            elif fingers == [1, 0, 0, 0, 0]:
                ImgCanvas = np.zeros_like(board)

    # Convert to grayscale and invert image for proper drawing overlay
    imgGray = cv2.cvtColor(ImgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 30, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # Ensure better blending of ImgCanvas and board
    board = cv2.bitwise_and(board, imgInv)
    board = cv2.bitwise_or(board, ImgCanvas)

    # Add header on top
    board[0:125, 0:1280] = header

    cv2.imshow('Frame', board) # Actual Notebook
    cv2.imshow("Canvas", ImgCanvas) # Canvas (black and white)
    cv2.imshow("Inv", imgInv) # Inverse - Canvas (white and black)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()