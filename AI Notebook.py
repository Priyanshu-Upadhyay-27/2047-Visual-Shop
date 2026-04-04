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

graph_mode = False

# Voice navigation setup
model_path = "C:/Users/priya/Desktop/Speech_Processing/vosk-model-small-en-us-0.15"
speech_model = Model(model_path)
recognizer = KaldiRecognizer(speech_model, 16000, '["blue", "yellow", "red", "green", "eraser", "clear screen", "exit", "[unk]", "graph", "normal"]')

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
            print(f"Recognized text:{recognized_text}")

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
def sendToAI(model, ImgCanvas , Lmlist, thumb_ring):
    #print(thumb_ring)
    if (thumb_ring < 30 and Lmlist[20][2] < Lmlist[18][2]) and not Lmlist[20][2] > Lmlist[18][2]:
        pil_image = Image.fromarray(ImgCanvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        print(response.text)
        return response.text

# Function to display graph
def display_graph(board):
    global graph_image
    if graph_image is not None:
        # Resize the graph to fit a specific portion of the screen (optional)
        board[120:598, 751:1269] = graph_image[0:478, 0:519]  # Only crop based on graph's original size
    else:
        print("Graph image not found!")
def finger_pos(fingers):
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
    return fingers

def preprocess_frame(frame):
    """
    Preprocesses the input frame for robust hand detection, applying contrast enhancement,
    denoising, brightness adjustment, and optional skin segmentation.

    Args:
        frame (numpy.ndarray): The input frame from the camera.

    Returns:
        numpy.ndarray: The preprocessed frame ready for hand detection.
    """
    # Step 1: Resize for consistency (optional)
    frame = cv2.resize(frame, (640, 480))

    # Step 2: Brightness and Contrast Boost (for dark lighting)
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    bright_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Step 3: Convert to YCrCb and apply CLAHE on the Y channel
    ycrcb = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    # Merge channels back
    ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    enhanced_frame = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    # Step 4: Apply Bilateral Filter (preserves edges)
    filtered_frame = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)

    # Step 5: Skin Segmentation (Optional fallback or visualization)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    skin_mask_morph = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    skin_segmented = cv2.bitwise_and(frame, frame, mask=skin_mask_morph)
    # Optional visualization for debugging
    # cv2.imshow("Original", frame)
    # cv2.imshow("Brightened", bright_frame)
    # cv2.imshow("CLAHE Enhanced", enhanced_frame)
    # cv2.imshow("Filtered", filtered_frame)
    # cv2.imshow("Skin Segmented", skin_segmented)
    return skin_segmented


# Some important initialization
# displaying the canvas at the header
# Tool Bar
folderPath1 = "Tool Bar"
mylist1 = os.listdir(folderPath1)
overLayList1 = []
for imPath1 in mylist1:
    image1 = cv2.imread(f"{folderPath1}/{imPath1}")
    overLayList1.append(image1)
bar = overLayList1[0]

# Color Palette
folderPath2 = "Color Palette"
mylist2 = os.listdir(folderPath2)
overLayList2 = []
for imPath2 in mylist2:
    image2 = cv2.imread(f"{folderPath2}/{imPath2}")
    overLayList2.append(image2)
palette = overLayList2[0]

# Top Header : Phantom Board and toolbar button option and graphs also
folderPath3 = "Extras"
mylist3 = os.listdir(folderPath3)
overLayList3 = []
for imPath3 in mylist3:
    image3 = cv2.imread(f"{folderPath3}/{imPath3}")
    overLayList3.append(image3)
button = overLayList3[2]
graph_image = overLayList3[1]
header = overLayList3[0]

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
penThickness = 15

while True:
    success, board = cap.read()
    board = cv2.flip(board, 1)
    board_p = preprocess_frame(board)
    boardRGB = cv2.cvtColor(board_p, cv2.COLOR_BGR2RGB)
    results = hands.process(boardRGB) # processed result
    Lmlist = [] #initializing empty list, which will contain coordinates of landmarks
    if graph_mode:
        board = np.ones_like(board) * 255  # White canvas
    else:
        pass

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
            thumb_index = math.hypot(x3-x1, y3-y1)
            # appending 1 to fingers[] because we will be denoting 1 as open finger and vice versa for 0.
            # function creating a list of 1 and 0, where 1 and 0 are standing and resting pos.
            fingers = [] # initializing an empty list
            # Commands are provided according to the fingers position
            fingers = finger_pos(fingers)

            sendToAI(model, ImgCanvas, Lmlist, thumb_ring) # Model is called
            # Commands are provided according to the fingers position
            # Selection Mode
            # Detect click on the tool button
            # Tool button interaction

            # Perform selection if an option is clicked
            if fingers == [0, 1, 1, 0, 0]:  # Index finger is up (selection mode)
                if 43 < x1 < 93:  # Palette column
                    if 207 < y1 < 257:  # Red
                        palette = overLayList2[1]
                        drawColor = (0, 0, 255)
                        penThickness = 15
                    elif 291 < y1 < 341:  # Green
                        palette = overLayList2[2]
                        drawColor = (0, 255, 0)
                        penThickness = 15
                    elif 375 < y1 < 425:  # Blue
                        palette = overLayList2[3]
                        drawColor = (255, 0, 0)
                        penThickness = 15
                    elif 459 < y1 < 509:  # Yellow
                        palette = overLayList2[4]
                        drawColor = (0, 255, 255)
                        penThickness = 15
                    elif 543 < y1 < 593:  # Pen size
                        palette = overLayList2[5]
                        penThickness = 30
                    elif 627 < y1 < 677:  # Eraser
                        palette = overLayList2[7]
                        drawColor = (0, 0, 0)  # Set color to black for eraser
                        penThickness = 50

                # Toolbar selections
                elif 76 < y1 < 147:  # Toolbar row
                    if 408 < x1 < 510:  # Undo
                        bar = overLayList1[1]
                        print("Undo action triggered.")
                    elif 542 < x1 < 644:  # Save
                        bar = overLayList1[2]
                        cv2.imwrite("Saved_canvas.png", board)
                        print("Current Canvas Saved")
                    elif 666 < x1 < 768:  # Voice Navigation
                        bar = overLayList1[3]
                        print("Voice Navigation Triggered.")
                        recognized_command = recognize_speech()
                        if recognized_command == "undo":
                            print("Undo command recognized.")
                        elif recognized_command == "save":
                            print("Save command recognized.")
                        elif recognized_command == "clear screen":
                            ImgCanvas = np.zeros_like(board)
                        elif recognized_command == "blue":
                            palette = overLayList2[3]
                            drawColor = (250, 0, 0)  # Blue
                            penThickness = 15
                        elif recognized_command == "yellow":
                            palette = overLayList2[4]
                            drawColor = (0, 255, 255)  # Yellow
                            penThickness = 15
                        elif recognized_command == "green":
                            palette = overLayList2[2]
                            drawColor = (0, 255, 0)  # Green
                            penThickness = 15
                        elif recognized_command == "red":
                            palette = overLayList2[1]
                            drawColor = (0, 0, 255)  # Red
                            penThickness = 15
                        elif recognized_command == "eraser":
                            palette = overLayList2[7]
                            drawColor = (0, 0, 0)  # Black
                            penThickness = 50
                        elif recognized_command == "graph":
                            graph_mode = True
                            print("Blank Canvas")
                        elif recognized_command == "normal": #getting back to normal webcam
                            graph_mode = False
                        elif recognized_command == "exit":
                            break
                        else:
                            print("Command not recognized")
                    elif 788 < x1 < 890:  # Blank Canvas
                        ar = overLayList1[4]  # Optional: update overlay if you have a graph icon
                        graph_mode = True
                        print("Blank Canvas")

                cv2.rectangle(board, (x1, y1 - 25), (x2, y2 + 25), drawColor, -1)#finger pointer
                xp, yp = x1, y1 # re - arrangement
        # Writting Mode
            elif fingers == [0,1,0,0,0] and not (fingers[2] and fingers[3] and fingers[4]):  # Index finger is up
                cv2.circle(board, (x1, y1), 15, drawColor, -1)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(board, (xp, yp), (x1, y1), drawColor, penThickness)
                cv2.line(ImgCanvas, (xp, yp), (x1, y1), drawColor, penThickness)
                xp, yp = x1, y1
            elif fingers == [1,1,1,1,0]:
                x = sendToAI(model, ImgCanvas, fingers, thumb_ring)
                cv2.putText(board, x, (900, 400), cv2.FONT_HERSHEY_PLAIN, 20, drawColor, 2)
            # clearing all
            elif fingers == [0, 0, 0, 0, 1]:
                ImgCanvas = np.zeros_like(board)

    # Convert to grayscale and invert image for proper drawing overlay
    imgGray = cv2.cvtColor(ImgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 30, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # Ensure better blending of ImgCanvas and board
    board = cv2.bitwise_and(board, imgInv)
    board = cv2.bitwise_or(board, ImgCanvas)

    # Resize header to match the target region's dimensions
    header_resized = cv2.resize(header, (1280, 66))

    # Add header on top
    board[0:66, 0:1280] = header # header: Phantom Board
    board[186:693, 25:111] = palette # color palette
    board[76:151, 388:892] = bar # Tool Bar
    board[76:162, 25:111] = button #tool button

    # Display color palette and toolbar only when panels_open is True

    cv2.imshow('Frame', board) # Actual Notebook
    cv2.imshow("Canvas", ImgCanvas) # Canvas (black and white)
    cv2.imshow("Inv", imgInv) # Inverse - Canvas (white and black)

    if cv2.waitKey(1) == ord("q"):
        break

    #elif cv2.waitKey(1) == ord("z"):
        #undo()  # Trigger undo when "z" is pressed

cap.release()
cv2.destroyAllWindows()
