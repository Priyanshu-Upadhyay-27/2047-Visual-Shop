# 🎨 Vision Board: AI-Powered Virtual Notebook

Vision Board is an interactive, touchless virtual whiteboard powered by **Computer Vision, Voice Recognition, and Generative AI**. It enables users to draw in the air using hand gestures, control tools via voice, and even solve handwritten math problems using Google's Gemini AI.

---

## ✨ Key Features

* 🤚 **Touchless Air Canvas**
  Draw in real-time using your index finger powered by MediaPipe hand tracking.

* 🤖 **AI Math Solver**
  Write a math problem, trigger a gesture, and let **Gemini 1.5 Flash** interpret and solve it.

* 🎙️ **Voice Navigation**
  Use offline voice commands (Vosk) to switch tools, colors, and modes.

* 🎨 **Dynamic UI Elements**
  Interactive toolbar and color palette for seamless tool switching.

* 📈 **Multiple Canvas Modes**
  Toggle between a plain whiteboard and graph paper overlay.

* 📸 **Save & Undo**
  Save your work or undo recent actions easily.

---

## 🛠️ Tech Stack & Dependencies

### Computer Vision

* `opencv-python`
* `mediapipe`
* `Pillow`

### Generative AI

* `google-generativeai`

### Voice Recognition

* `vosk`
* `pyaudio`
* `noisereduce`

### Math & Utilities

* `numpy`
* `math`

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Priyanshu-Upadhyay-27/2047-Visual-Shop/tree/new_updates
```

### 2. Install Dependencies

```bash
pip install opencv-python mediapipe numpy google-generativeai pillow pyaudio noisereduce vosk
```

### 3. Download Vosk Voice Model

* Download: `vosk-model-small-en-us-0.15`
* Extract it
* Update `model_path` in `AI Notebook.py`

---

### 4. Configure Gemini API Key

* Get API key from **Google AI Studio**

⚠️ **Important (Don’t Ignore This):**

```python
import os
API_KEY = os.getenv("GEMINI_API_KEY")
```

---

## 🕹️ How to Use

### Run the Application

```bash
python "AI Notebook.py"
```

---

## ✋ Hand Gestures

* **Drawing Mode:** 👉 Index Finger Up
* **Selection Mode:** 👉 Index + Middle Finger Up
* **Clear Screen:** 👉 Pinky Finger Up
* **AI Solve Mode:** 👉 Thumb + Ring Finger Together

---

## 🗣️ Voice Commands

| Command      | Action            |
| ------------ | ----------------- |
| blue/red/etc | Change pen color  |
| eraser       | Switch to eraser  |
| clear screen | Clear canvas      |
| graph        | Enable graph mode |
| normal       | Default canvas    |
| exit         | Close application |

---

## 📁 Project Structure

```
.
├── AI Notebook.py              # Main application
├── OrganizedCode.py           # Refactored OOP version
├── AI Notebook_older_version.py
├── Color Palette/             # UI assets
├── Tool Bar/                  # Toolbar assets
├── Header/                    # Header graphics
├── Extras/                    # Logos, overlays, etc.
```

---

## ⚠️ Troubleshooting

### PyAudio Issues (Windows)

Use precompiled `.whl` instead of pip install.

### Voice Not Working

* Check microphone access
* Verify correct `model_path`

### Poor Hand Tracking

* Improve lighting conditions
* Use natural light if possible
* Ensure camera clarity

---

## 🚧 Future Improvements

* Replace gesture logic with **ML-based gesture classification**
* Add **multi-user support**
* Improve **latency + performance optimization**
* Convert script into **modular architecture (Flask / FastAPI)**
* Integrate **cloud-based inference**

---

