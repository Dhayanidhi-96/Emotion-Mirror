# 🪞 Emotion Mirror

A real-time facial emotion detection app that uses your webcam to predict emotions — happy, sad, angry, surprised, fearful, disgusted, or neutral — and displays them live on screen with a clean overlay.

---

## 📸 Demo

> See `demo.mp4` in the repository root for a 1–3 minute screen recording of the app in action.

---

## ✨ Features

- 🎥 Real-time webcam feed with live emotion detection
- 🧠 Detects emotions using a pre-trained deep learning model (no training required)
- 👥 Supports **multiple faces** in the same frame — each gets its own label
- 😶 Gracefully handles **no face detected** with an on-screen message
- 🎨 Clean, color-coded bounding boxes and emotion labels per face

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| [uv](https://github.com/astral-sh/uv) | Fast Python package manager |
| [OpenCV](https://opencv.org/) | Webcam capture and frame drawing |
| [FER](https://github.com/justinshenk/fer) | Pre-trained facial emotion recognition |
| TensorFlow | Backend for FER's mini-XCEPTION CNN |
| NumPy | Frame array processing |

---

## 📁 Project Structure

```
emotion-mirror/
├── src/
│   └── emotion_mirror/
│       ├── __init__.py      # package init
│       ├── main.py          # entry point — starts webcam loop
│       ├── detector.py      # FER emotion detection logic
│       └── display.py       # draws boxes and labels on frame
├── README.md
├── LICENSE
├── pyproject.toml           # project config and dependencies
└── .gitignore
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) installed — if not, install it:
  ```bash
  pip install uv
  ```
- A working webcam

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Dhayanidhi-96/Emotion-Mirror.git
cd Emotion-Mirror
```

---

### Step 2 — Create Virtual Environment

```bash
uv venv --python 3.11
```

Activate it:

- **Windows:**
  ```powershell
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

---

### Step 3 — Install Dependencies

```bash
uv pip install -e .
```

> **Note:** On first run, the FER library will download pre-trained model weights (~5 MB). This only happens once.

---

### Step 4 — Run the App

```bash
python src/emotion_mirror/main.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit the app |

---

## 🧪 How It Works

```
Webcam Frame
     ↓
OpenCV captures and mirrors the frame
     ↓
MTCNN detects face region(s)
     ↓
Pre-trained mini-XCEPTION model runs inference
     ↓
Outputs emotion probabilities:
  happy: 82%, neutral: 10%, sad: 5%, angry: 3% ...
     ↓
Top emotion + confidence drawn on screen
```

The **FER** library uses a **mini-XCEPTION** convolutional neural network trained on the **FER2013 dataset** — a dataset of ~35,000 grayscale face images labeled across 7 emotion categories. We use this model purely for **inference** (no training, no fine-tuning).

---

## 🔲 Edge Case Handling

| Situation | Behavior |
|---|---|
| No face in frame | Displays `"No Face Detected"` message on screen |
| Multiple faces | Each face gets its own bounding box and emotion label |
| Small / false detections | Filtered out automatically by minimum face area threshold |
| Low lighting / partial face | MTCNN gracefully skips detection, no crash |

---

## 💡 What I Learned / Found Challenging

Building Emotion Mirror was both a technical and creative challenge.

The most interesting part was understanding how pre-trained models work in practice — rather than training from scratch, we rely on a model already trained on thousands of labeled face images and simply run inference on live webcam frames. This made me appreciate how much of modern AI development is about **integrating and orchestrating** existing tools intelligently, not just building models.

The most challenging part was making the experience feel **polished and real-time**. Raw OpenCV windows can feel very utilitarian. Getting the bounding boxes and emotion labels to render cleanly at a good frame rate — while also handling edge cases like no face or multiple faces gracefully — required careful thought about the rendering loop and how to avoid lag.

I also found it interesting to think about the **limitations** of emotion detection: lighting conditions, partial faces, and the inherent ambiguity of human expression all affect accuracy. Using MTCNN over Haar Cascades for face detection significantly improved reliability in real-world conditions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of the Bipolar Factory — Emotion Mirror assignment.*
