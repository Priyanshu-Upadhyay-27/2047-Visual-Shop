import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import math
import pyaudio
import json
import noisereduce as nr
from vosk import Model, KaldiRecognizer

# Voice navigation setup
model_path = "C:/Users/priya/Desktop/Speech_Processing/vosk-model-small-en-us-0.15"  # Replace with the correct path
speech_model = Model(model_path)
recognizer = KaldiRecognizer(speech_model, 16000, '["blue", "yellow", "red", "green", "eraser", "clear screen", "exit", "[unk]"]')

def recognize_speech():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000)
    stream.start_stream()
    print("Listening for speech...")

    while True:
        data = stream.read(4000)
        audio_data = np.frombuffer(data, dtype=np.int16)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=16000)

        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_json = json.loads(result)
            recognized_text = result_json.get('text')
            print(f"Recognized text: {recognized_text}")

            if "exit" in recognized_text.lower():
                print("Exiting program...")
                stream.stop_stream()
                stream.close()
                p.terminate()
                return "exit"
            return recognized_text.lower()

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

def preprocess_frame(frame):
    """
    Preprocesses the input frame for robust hand detection, using YCbCr color space and CLAHE for illumination correction.

    Args:
        frame (numpy.ndarray): The input frame from the camera.

    Returns:
        numpy.ndarray: The preprocessed frame ready for hand detection.
    """
    # Step 1: Convert to YCbCr color space (less sensitive to lighting)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Step 2: Skin Segmentation in YCbCr color space
    # Define lower and upper thresholds for skin color in YCbCr
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Create skin mask
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_frame = clahe.apply(gray)

    # Step 4: Noise Reduction using bilateral filter
    blurred_frame = cv2.bilateralFilter(normalized_frame, 9, 75, 75)

    # Step 5: Apply morphological operations to remove noise and improve segmentation
    skin_mask_morph = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Step 6: Skin Segmentation
    skin_segmented = cv2.bitwise_and(frame, frame, mask=skin_mask_morph)

    # Step 7: Visualization (Optional)
    cv2.imshow("Skin Mask", skin_mask)
    cv2.imshow("Morphological Skin Mask", skin_mask_morph)
    cv2.imshow("Skin Segmentation", skin_segmented)

    return skin_segmented

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
    board_p = preprocess_frame(board)
    boardRGB = cv2.cvtColor(board_p, cv2.COLOR_BGR2RGB)
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
            x5, y5 = Lmlist[20][1], Lmlist[20][2]
            thumb_ring = math.hypot(x4 - x3, y4 - y3)
            index_middle = math.hypot(x2 - x1, y2 - y1)
            thumb_pinky = math.hypot(x5 - x3, y5 - y3)
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

            if thumb_pinky < 25:
                vocal = recognize_speech()
                if vocal == "blue":
                    header = overLayList[0]
                    drawColor = (250, 0, 0)  # Blue
                    penThickness = 15
                elif vocal == "yellow":
                    header = overLayList[1]
                    drawColor = (0, 255, 255)  # Yellow
                    penThickness = 15
                elif vocal == "green":
                    header = overLayList[2]
                    drawColor = (0, 255, 0)  # Green
                    penThickness = 15
                elif vocal == "red":
                    header = overLayList[3]
                    drawColor = (0, 0, 255)  # Red
                    penThickness = 15
                elif vocal == "eraser":
                    header = overLayList[4]
                    drawColor = (0, 0, 0)  # Black
                    penThickness = 50
                elif vocal == "clear screen":
                    ImgCanvas = np.zeros_like(board)
            # Commands are provided according to the fingers position
            # Selection Mode
            if (fingers[1] and fingers[2] and not (fingers[3] and fingers[4])):
                #setting up of making header dynamic using different canvas
                if y1 < 125:
                    if 0 < x1 < 140 == "red":
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