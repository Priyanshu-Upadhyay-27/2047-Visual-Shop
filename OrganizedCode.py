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


class HandTracker:
    """Handles hand detection and tracking using MediaPipe."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.8):
        """Initialize the MediaPipe hand tracking components."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        """
        Detect hands in the given frame.

        Args:
            frame: The input frame from the camera
            draw: Whether to draw hand landmarks on the frame

        Returns:
            The processed frame with hand landmarks (if draw=True)
            Results object from MediaPipe
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return frame, results

    def find_position(self, frame, results):
        """
        Find position of hand landmarks.

        Args:
            frame: The input frame
            results: Results from MediaPipe hand detection

        Returns:
            landmarks_list: List of landmark positions
            hand_label: "Left" or "Right" hand label
        """
        h, w, _ = frame.shape
        landmarks_list = []
        hand_label = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_list.append([id, cx, cy])

            # Determine hand label
            if results.multi_handedness:
                for handedness in results.multi_handedness:
                    hand_label = handedness.classification[0].label

        return landmarks_list, hand_label

    def get_finger_status(self, landmarks_list, hand_label):
        """
        Determine which fingers are up.

        Args:
            landmarks_list: List of landmark positions
            hand_label: "Left" or "Right" hand label

        Returns:
            List where each element represents if a finger is up (1) or down (0)
        """
        if not landmarks_list:
            return []

        fingers = []

        # Thumb
        if hand_label == "Left" and landmarks_list[4][1] > landmarks_list[2][1]:
            fingers.append(1)
        elif hand_label == "Right" and landmarks_list[4][1] < landmarks_list[2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in [8, 12, 16, 20]:
            if landmarks_list[id][2] < landmarks_list[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


class ImageProcessor:
    """Handles image processing operations for enhanced hand tracking."""

    @staticmethod
    def preprocess_frame(frame):
        """
        Preprocesses the input frame for robust hand detection.

        Args:
            frame: The input frame from the camera

        Returns:
            The preprocessed frame ready for hand detection
        """
        # Resize for consistency
        frame = cv2.resize(frame, (640, 480))

        # Brightness and Contrast Boost
        alpha = 1.2  # Contrast control
        beta = 30  # Brightness control
        bright_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Convert to YCrCb and apply CLAHE on the Y channel
        ycrcb = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)

        # Merge channels back
        ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
        enhanced_frame = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

        # Apply Bilateral Filter (preserves edges)
        filtered_frame = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)

        # Skin Segmentation
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        skin_mask_morph = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        skin_segmented = cv2.bitwise_and(frame, frame, mask=skin_mask_morph)

        return skin_segmented


class VoiceRecognizer:
    """Handles voice command recognition."""

    def __init__(self, model_path):
        """
        Initialize voice recognition components.

        Args:
            model_path: Path to the Vosk model directory
        """
        try:
            self.speech_model = Model(model_path)
            self.recognizer = KaldiRecognizer(
                self.speech_model,
                16000,
                '["blue", "yellow", "red", "green", "eraser", "clear screen", "exit", "[unk]", "graph", "normal", "undo", "save"]'
            )
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing voice recognizer: {e}")
            self.speech_model = None

    def recognize(self):
        """
        Listen for and recognize voice commands.

        Returns:
            The recognized command text or None if recognition failed
        """
        if not self.speech_model:
            print("Voice recognition model not available")
            return None

        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000
            )
            stream.start_stream()
            print("Listening for speech...")

            while True:
                data = stream.read(4000)
                audio_data = np.frombuffer(data, dtype=np.int16)
                reduced_noise = nr.reduce_noise(y=audio_data, sr=16000)

                if self.recognizer.AcceptWaveform(data):
                    result = self.recognizer.Result()
                    result_json = json.loads(result)
                    recognized_text = result_json.get('text')
                    print(f"Recognized text: {recognized_text}")

                    if "exit" in recognized_text.lower():
                        print("Exiting program...")
                        stream.stop_stream()
                        stream.close()
                        return "exit"

                    stream.stop_stream()
                    stream.close()
                    return recognized_text.lower()
        except Exception as e:
            print(f"Error in voice recognition: {e}")
            return None


class AIHelper:
    """Handles AI model integration for interpreting drawn content."""

    def __init__(self, api_key):
        """
        Initialize the AI model.

        Args:
            api_key: API key for accessing the generative AI service
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print(f"Error initializing AI helper: {e}")
            self.model = None

    def interpret_drawing(self, canvas_image, landmarks_list, thumb_ring_distance):
        """
        Send the drawing to the AI model for interpretation.

        Args:
            canvas_image: The canvas image containing the drawing
            landmarks_list: List of hand landmark positions
            thumb_ring_distance: Distance between thumb and ring finger

        Returns:
            The interpreted text from the AI model
        """
        if not self.model:
            return "AI model not available"

        try:
            # Only interpret if specific gesture is detected
            if (thumb_ring_distance < 30 and
                    landmarks_list[20][2] < landmarks_list[18][2] and
                    not landmarks_list[20][2] > landmarks_list[18][2]):
                pil_image = Image.fromarray(canvas_image)
                response = self.model.generate_content(["Solve this math problem", pil_image])
                return response.text
            return None
        except Exception as e:
            print(f"Error in AI interpretation: {e}")
            return f"Error: {str(e)}"


class ResourceLoader:
    """Handles loading of image resources used in the application."""

    def __init__(self):
        """Initialize resource storage."""
        self.tool_bar_images = []
        self.palette_images = []
        self.extra_images = []
        self.header = None
        self.button = None
        self.graph_image = None

    def load_resources(self):
        """
        Load all resource images.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load tool bar images
            self.tool_bar_images = self._load_images_from_folder("Tool Bar")

            # Load color palette images
            self.palette_images = self._load_images_from_folder("Color Palette")

            # Load extra images
            self.extra_images = self._load_images_from_folder("Extras")

            if self.extra_images:
                self.header = self.extra_images[0]
                #self.graph_image = self.extra_images[1]
                self.button = self.extra_images[2]

            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False

    def _load_images_from_folder(self, folder_path):
        """
        Load all images from a specified folder.

        Args:
            folder_path: Path to the folder containing images

        Returns:
            List of loaded images
        """
        images = []
        try:
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
        except Exception as e:
            print(f"Error loading images from {folder_path}: {e}")
        return images


class DrawingCanvas:
    """Manages the drawing canvas and drawing operations."""

    def __init__(self, width=1280, height=720):
        """
        Initialize the drawing canvas.

        Args:
            width: Width of the canvas
            height: Height of the canvas
        """
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self.draw_color = (250, 250, 0)  # Default color
        self.pen_thickness = 15
        self.prev_x = 0
        self.prev_y = 0
        self.graph_mode = False

    def clear(self):
        """Clear the canvas."""
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)

    def draw_line(self, x1, y1):
        """
        Draw a line from previous point to current point.

        Args:
            x1: Current x-coordinate
            y1: Current y-coordinate
        """
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x1, y1

        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1),
                 self.draw_color, self.pen_thickness)
        self.prev_x, self.prev_y = x1, y1

    def reset_prev_point(self):
        """Reset the previous point."""
        self.prev_x, self.prev_y = 0, 0

    def set_color(self, color):
        """
        Set the drawing color.

        Args:
            color: RGB color tuple
        """
        self.draw_color = color

    def set_thickness(self, thickness):
        """
        Set the pen thickness.

        Args:
            thickness: Pen thickness value
        """
        self.pen_thickness = thickness

    def toggle_graph_mode(self, mode):
        """
        Toggle graph mode.

        Args:
            mode: Boolean indicating if graph mode should be enabled
        """
        self.graph_mode = mode

    def get_display_image(self):
        """
        Get the canvas for display.

        Returns:
            The processed canvas image for display
        """
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        return img_inv

    def save(self, filename="Saved_canvas.png"):
        """
        Save the canvas to a file.

        Args:
            filename: Name of the file to save

        Returns:
            True if successful, False otherwise
        """
        try:
            cv2.imwrite(filename, self.canvas)
            return True
        except Exception as e:
            print(f"Error saving canvas: {e}")
            return False


class PhantomBoard:
    """Main application class that integrates all components."""

    def __init__(self, config=None):
        """
        Initialize the application with optional configuration.

        Args:
            config: Configuration dictionary with settings
        """
        # Default configuration
        self.config = {
            'model_path': "C:/Users/priya/Desktop/Speech_Processing/vosk-model-small-en-us-0.15",
            'api_key': "AIzaSyC2YBSBArq1UxYk456q6FQ9mDoA7OoYrKc",
            'camera_width': 1280,
            'camera_height': 720
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # Initialize components
        self.hand_tracker = HandTracker()
        self.image_processor = ImageProcessor()
        self.voice_recognizer = VoiceRecognizer(self.config['model_path'])
        self.ai_helper = AIHelper(self.config['api_key'])
        self.resources = ResourceLoader()
        self.canvas = DrawingCanvas(self.config['camera_width'], self.config['camera_height'])

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.config['camera_width'])
        self.cap.set(4, self.config['camera_height'])

        # UI state variables
        self.current_palette = None
        self.current_bar = None
        self.running = True

    def initialize(self):
        """
        Initialize resources and prepare the application.

        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.resources.load_resources():
            print("Failed to load resources")
            return False

        self.current_palette = self.resources.palette_images[0]
        self.current_bar = self.resources.tool_bar_images[0]
        return True

    def process_selection(self, x1, y1, fingers):
        """
        Process selection gestures.

        Args:
            x1: x-coordinate of the index finger
            y1: y-coordinate of the index finger
            fingers: List of finger statuses
        """
        # Check if in selection mode (index and middle finger up)
        if fingers == [0, 1, 1, 0, 0]:
            # Palette column selection
            if 43 < x1 < 93:
                if 207 < y1 < 257:  # Red
                    self.current_palette = self.resources.palette_images[1]
                    self.canvas.set_color((0, 0, 255))
                    self.canvas.set_thickness(15)
                elif 291 < y1 < 341:  # Green
                    self.current_palette = self.resources.palette_images[2]
                    self.canvas.set_color((0, 255, 0))
                    self.canvas.set_thickness(15)
                elif 375 < y1 < 425:  # Blue
                    self.current_palette = self.resources.palette_images[3]
                    self.canvas.set_color((255, 0, 0))
                    self.canvas.set_thickness(15)
                elif 459 < y1 < 509:  # Yellow
                    self.current_palette = self.resources.palette_images[4]
                    self.canvas.set_color((0, 255, 255))
                    self.canvas.set_thickness(15)
                elif 543 < y1 < 593:  # Pen size
                    self.current_palette = self.resources.palette_images[5]
                    self.canvas.set_thickness(30)
                elif 627 < y1 < 677:  # Eraser
                    self.current_palette = self.resources.palette_images[7]
                    self.canvas.set_color((0, 0, 0))
                    self.canvas.set_thickness(50)

            # Toolbar row selection
            elif 76 < y1 < 147:
                if 408 < x1 < 510:  # Undo
                    self.current_bar = self.resources.tool_bar_images[1]
                    print("Undo action triggered.")
                elif 542 < x1 < 644:  # Save
                    self.current_bar = self.resources.tool_bar_images[2]
                    self.canvas.save()
                    print("Current Canvas Saved")
                elif 666 < x1 < 768:  # Voice Navigation
                    self.current_bar = self.resources.tool_bar_images[3]
                    print("Voice Navigation Triggered.")
                    self.process_voice_command()
                elif 788 < x1 < 890:  # Blank Canvas / Normal Toggle
                    if self.canvas.graph_mode:
                        # If already in graph mode, switch to normal mode
                        self.current_bar = self.resources.tool_bar_images[0]  # Reset to default toolbar
                        self.canvas.toggle_graph_mode(False)
                        print("Switched to Normal Mode")
                    else:
                        # If in normal mode, switch to graph mode
                        self.current_bar = self.resources.tool_bar_images[4]
                        self.canvas.toggle_graph_mode(True)
                        print("Switched to Blank Canvas")

    def process_voice_command(self):
        """Process voice commands and take appropriate actions."""
        recognized_command = self.voice_recognizer.recognize()

        if not recognized_command:
            return

        if recognized_command == "undo":
            print("Undo command recognized.")
        elif recognized_command == "save":
            self.canvas.save()
            print("Save command recognized.")
        elif recognized_command == "clear screen":
            self.canvas.clear()
        elif recognized_command == "blue":
            self.current_palette = self.resources.palette_images[3]
            self.canvas.set_color((250, 0, 0))
        elif recognized_command == "yellow":
            self.current_palette = self.resources.palette_images[4]
            self.canvas.set_color((0, 255, 255))
        elif recognized_command == "green":
            self.current_palette = self.resources.palette_images[2]
            self.canvas.set_color((0, 255, 0))
        elif recognized_command == "red":
            self.current_palette = self.resources.palette_images[1]
            self.canvas.set_color((0, 0, 255))
        elif recognized_command == "eraser":
            self.current_palette = self.resources.palette_images[7]
            self.canvas.set_color((0, 0, 0))
            self.canvas.set_thickness(50)
        elif recognized_command == "graph":
            self.canvas.toggle_graph_mode(True)
            print("Blank Canvas")
        elif recognized_command == "normal":
            self.canvas.toggle_graph_mode(False)
        elif recognized_command == "exit":
            self.running = False
        else:
            print("Command not recognized")

    def process_drawing(self, landmarks_list, fingers):
        """
        Process drawing gestures.

        Args:
            landmarks_list: List of hand landmark positions
            fingers: List of finger statuses
        """
        if len(landmarks_list) == 0:
            return

        x1, y1 = landmarks_list[8][1], landmarks_list[8][2]
        x2, y2 = landmarks_list[12][1], landmarks_list[12][2]
        x3, y3 = landmarks_list[4][1], landmarks_list[4][2]
        x4, y4 = landmarks_list[16][1], landmarks_list[16][2]

        # Calculate distances
        thumb_ring = math.hypot(x4 - x3, y4 - y3)
        thumb_index = math.hypot(x3 - x1, y3 - y1)

        # Check AI interpretation
        ai_result = self.ai_helper.interpret_drawing(self.canvas.canvas, landmarks_list, thumb_ring)
        if ai_result:
            print("AI Interpretation:", ai_result)

        # Drawing mode (only index finger up)
        if fingers == [0, 1, 0, 0, 0]:
            self.canvas.draw_line(x1, y1)
        # AI interpretation mode
        elif fingers == [1, 1, 1, 1, 0]:
            pass
        # Clear canvas (pinky finger up)
        elif fingers == [0, 0, 0, 0, 1]:
            self.canvas.clear()
        else:
            self.canvas.reset_prev_point()

    def compose_frame(self, board):
        """
        Compose the final frame with UI elements.

        Args:
            board: The base board frame

        Returns:
            The composed frame ready for display
        """
        # Get the inverted canvas for blending
        img_inv = self.canvas.get_display_image()

        # Blend the canvas with the board
        board = cv2.bitwise_and(board, img_inv)
        board = cv2.bitwise_or(board, self.canvas.canvas)

        # Add UI elements
        board[0:66, 0:1280] = self.resources.header  # Header
        board[186:693, 25:111] = self.current_palette  # Color palette
        board[76:151, 388:892] = self.current_bar  # Tool bar
        board[76:162, 25:111] = self.resources.button  # Tool button

        return board

    def run(self):
        """Run the main application loop."""
        if not self.initialize():
            print("Initialization failed")
            return

        while self.running:
            success, board = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break

            # Flip frame horizontally for mirror effect
            board = cv2.flip(board, 1)

            # Preprocess frame for better hand detection
            processed_board = self.image_processor.preprocess_frame(board)

            # Create blank board if in graph mode
            if self.canvas.graph_mode:
                board = np.ones_like(board) * 255  # White canvas

            # Detect hands
            board, results = self.hand_tracker.find_hands(board)

            # Get landmark positions and hand label
            landmarks_list, hand_label = self.hand_tracker.find_position(board, results)

            if landmarks_list:
                # Get finger status
                fingers = self.hand_tracker.get_finger_status(landmarks_list, hand_label)

                # Process selection gestures
                self.process_selection(landmarks_list[8][1], landmarks_list[8][2], fingers)

                # Process drawing gestures
                self.process_drawing(landmarks_list, fingers)

                # Draw finger pointer in selection mode
                if fingers == [0, 1, 1, 0, 0]:
                    x1, y1 = landmarks_list[8][1], landmarks_list[8][2]
                    x2, y2 = landmarks_list[12][1], landmarks_list[12][2]
                    cv2.rectangle(board, (x1, y1 - 25), (x2, y2 + 25), self.canvas.draw_color, -1)

            # Compose the final frame
            display_board = self.compose_frame(board)

            # Display frames
            cv2.imshow('Frame', display_board)
            cv2.imshow("Canvas", self.canvas.canvas)
            cv2.imshow("Inv", self.canvas.get_display_image())

            # Check for quit key
            if cv2.waitKey(1) == ord("q"):
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Optional configuration
    config = {
        'model_path': "C:/Users/priya/Desktop/Speech_Processing/vosk-model-small-en-us-0.15",
        'api_key': "AIzaSyC2YBSBArq1UxYk456q6FQ9mDoA7OoYrKc",
        'camera_width': 1280,
        'camera_height': 720
    }

    # Create and run the application
    app = PhantomBoard(config)
    app.run()