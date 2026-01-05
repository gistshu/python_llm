# Medical AI Transcription & Diarization System

[繁體中文版](README_zh-TW.md) | [English Version](README.md)

A local, privacy-focused automated transcription system designed for medical dialogues. This project combines OpenAI Whisper for ASR, Resemblyzer for speaker identification, and a modern web interface for correcting transcripts and generating training datasets for LLMs.

## 🌟 Key Features

*   **Secure Local Processing**: All audio processing happens locally (GPU/CPU), ensuring patient data privacy.
*   **High-Accuracy ASR**: Utilizes **OpenAI Whisper** for robust speech-to-text conversion.
*   **Speaker Identification**:
    *   Upload enrollment samples (e.g., `Doctor.wav`, `Patient.wav`) to automatically identify speakers.
    *   Uses **Resemblyzer** for voice embedding and cosine similarity matching.
    *   Confidence-based thresholding (default 0.7) for reliable naming.
*   **Smart Role Detection**: Automatically flags "Doctor" roles to structure the final dataset.
*   **Modern Web Interface (Flask)**:
    *   **Waveform Visualization**: Interactive player using `WaveSurfer.js` with zoom and timeline.
    *   **Bi-directional Sync**: Click transcript to seek audio; audio playback highlights text.
    *   **Editor**: `contenteditable` transcript for quick corrections.
    *   **Visual Indicators**: Color-coded badges for speakers (Black/Blue for Doctor, Red/Unknown for low confidence).
*   **LLM Dataset Builder**: Automates the creation of instruction-tuning datasets (`JSONL`) from processed dialogues.
*   **Interactive Chat Prototype**:
    *   **Virtual Doctor Avatar**: Professional AI-generated TCM professor character.
    *   **Functional Quick Action Buttons**: Recommend doctors, find nearby clinics, and health precautions.
    *   **Geolocation Integration**: Finds local TCM clinics using browser GPS.
*   **Optimized LoRA Training**:
    *   Dedicated scripts for **Phi-3** instruction tuning.
    *   **M2 Mac Support**: Specific optimizations for Apple Silicon (MPS acceleration).

## 🛠️ Tech Stack

*   **Backend**: Python, Flask, PyTorch
*   **ML Models**: OpenAI Whisper (ASR), Resemblyzer (Diarization), Phi-3 (LLM)
*   **LoRA Training**: PEFT (Parameter-Efficient Fine-Tuning)
*   **Frontend**: HTML5, Vanilla CSS (Medical Tech theme), JavaScript, WaveSurfer.js
*   **Audio**: Librosa, SoundFile, FFmpeg

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/medical-transcription-ai.git
    cd medical-transcription-ai
    ```

2.  **Install Dependencies**
    Requires Python 3.8+ and FFmpeg.
    ```bash
    pip install -r requirements.txt
    ```
    *Ensure `ffmpeg` is installed on your system (e.g., `brew install ffmpeg` on macOS).*

3.  **Environment Setup**
    Set the following environment variables if needed (or rely on defaults):
    ```bash
    export WHISPER_MODEL="small"      # base, small, medium, large-v2
    export DOCTOR_THRESHOLD="0.7"     # Confidence threshold for speaker ID
    export HUGGINGFACE_TOKEN="your_token" # If using protected models (optional)
    ```

## 🚀 Usage

### 1. Web Application (Interactive Mode)
Run the Flask server to access the UI.

```bash
python flaskv5.py
```

*   Open browser at `http://localhost:5000`
*   **Upload**: Select your conversation audio (`.wav`/`.mp3`) and optional enrollment samples (e.g., `Doctor_001.wav`).
*   **View & Edit**:
    *   Use the interactive waveform to navigate.
    *   Correct text directly in the transcript view.
    *   Save drafts and generate findings.
*   **Generate Dataset**: Click "Generate & Download" to get a clean `llm_dataset.jsonl`.

### 2. Batch Processing (CLI)
For processing multiple files at once using a configuration file or folder scan.

```bash
python batch_folder.py --input ./Case/dataset --output ./uploads/
```

### 3. LLM Training (LoRA)
Fine-tune models like Phi-3 using the generated datasets.

**For General CPU/GPU:**
```bash
python train/train_phi3.py --data_path ./datasets/my_data.jsonl --output_dir ./outputs_lora_v1
```

**For Apple Silicon (M1/M2):**
```bash
python train/train_phi3_m2.py --device mps --torch_dtype float16 --data_path ./datasets/my_data.jsonl --output_dir ./outputs_lora_v1
```

### 4. Interactive Chat (Inference)
Run the chat interface with your fine-tuned LoRA adapter.

```bash
python train/chat_web.py --lora_dir ./outputs_lora_v1 --force_traditional
```

## 📂 Project Structure

```
.
├── flaskv5.py          # Main Web Interface (Transcription & Diarization)
├── batch_folder.py     # Batch processing script
├── requirements.txt    # Python dependencies
├── train/              # LLM Training & Chat Inference
│   ├── train_phi3.py   # Phi-3 LoRA Trainer (CPU/Generic)
│   ├── train_phi3_m2.py# Phi-3 LoRA Trainer (Optimized for Apple M2)
│   ├── chat_web.py     # Chat UI Backend
│   └── templates/      # Chat UI HTML (chat_line.html)
├── uploads/            # Directory for processed jobs
└── Case/               # Example data and samples (Note: excluded from git)
```

## 🔍 How It Works

1.  **Preprocessing**: Audio is converted to 16kHz mono.
2.  **Transcription**: Whisper generates timestamps and text segments.
3.  **Diarization**:
    *   Extracts voice embeddings for each word/segment.
    *   Compares against uploaded "Enrollment" embeddings.
    *   Assigns speakers based on Cosine Similarity (> 0.7).
4.  **Dataset Generation**:
    *   Logic pairs "Patient" questions with "Doctor" advice.
    *   Outputs JSONL for fine-tuning Local LLMs (e.g., LLaMA, TaiwaLLM).

## 📝 License

[MIT License](LICENSE)
