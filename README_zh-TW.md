# 醫療 AI 語音轉錄與角色辨識系統 (Medical AI Transcription & Diarization System)

[English Version](README.md)

這是一個專為醫療對話設計的本地端、隱私導向自動轉錄系統。本專案結合 **OpenAI Whisper** 進行自動語音識別 (ASR)，使用 **Resemblyzer** 進行說話者識別，並提供現代化的網頁介面，方便使用者校對轉錄內容並產生 LLM 訓練資料集。

## 🌟 核心功能

*   **安全本地處理**：所有音訊處理皆在本地端 (GPU/CPU) 進行，確保病患資料隱私。
*   **高準確度 ASR**：利用 **OpenAI Whisper** 進行強大的語音轉文字功能。
*   **說話者識別 (Speaker Identification)**：
    *   上傳註冊樣本 (Enrollment samples)（例如：`Doctor.wav`, `Patient.wav`）以自動識別說話者。
    *   使用 **Resemblyzer** 提取聲紋特徵並進行餘弦相似度比對。
    *   基於信心閾值（預設 0.7）進行可靠的命名標記。
*   **智慧角色偵測**：自動標記 "Doctor" (醫師) 角色，以結構化最終產出的資料集。
*   **現代化網頁介面 (Flask)**：
    *   **波形視覺化**：整合 `WaveSurfer.js` 的互動式播放器，支援縮放與時間軸顯示。
    *   **雙向同步**：點擊文字可跳轉音訊；播放音訊時會自動高亮對應文字。
    *   **編輯器**：支援 `contenteditable` 的直覺式文字校對。
    *   **視覺指示器**：以顏色區分說話者標籤（黑色/藍色為醫師，紅色/Unknown 為低信心分數）。
*   **LLM 資料集建構器**：自動將處理後的對話轉換為指令微調 (Instruction-tuning) 資料集 (`JSONL`)。
*   **互動式對話原型 (Chat Prototype)**：
    *   **虛擬醫師頭像**：專業的 AI 生成中醫教授角色。
    *   **功能性快選按鈕**：提供「推薦醫師」、「最近院所」及「注意事項」快速功能。
    *   **地理定位整合**：支援瀏覽器 GPS 定位，快速搜尋附近的配合診所。
*   **優化 LoRA 訓練**：
    *   針對 **Phi-3** 模型設計的專用指令微調腳本。
    *   **M2 Mac 支援**：針對 Apple Silicon (MPS 加速) 進行記憶體與效能優化。

## 🛠️ 技術堆疊

*   **後端**：Python, Flask, PyTorch
*   **機器學習模型**：OpenAI Whisper (ASR), Resemblyzer (Diarization), Phi-3 (LLM)
*   **LoRA 訓練**：PEFT (Parameter-Efficient Fine-Tuning)
*   **前端**：HTML5, Vanilla CSS (醫療科技風格), JavaScript, WaveSurfer.js
*   **音訊處理**：Librosa, SoundFile, FFmpeg

## 📦 安裝說明

1.  **複製專案 (Clone Repository)**
    ```bash
    git clone https://github.com/yourusername/medical-transcription-ai.git
    cd medical-transcription-ai
    ```

2.  **安裝依賴套件**
    需要 Python 3.8+ 與 FFmpeg。
    ```bash
    pip install -r requirements.txt
    ```
    *請確保系統已安裝 `ffmpeg` (例如 macOS使用者可執行 `brew install ffmpeg`)。*

3.  **環境設定**
    如有需要，可設定以下環境變數（或使用預設值）：
    ```bash
    export WHISPER_MODEL="small"      # 可選 base, small, medium, large-v2
    export DOCTOR_THRESHOLD="0.7"     # 說話者識別的信心閾值
    export HUGGINGFACE_TOKEN="your_token" # 若使用受保護的模型 (選填)
    ```

## 🚀 使用方法

### 1. 網頁應用程式 (互動模式)
啟動 Flask 伺服器以進入圖形介面。

```bash
python flaskv5.py
```

*   開啟瀏覽器前往 `http://localhost:5000`
*   **上傳 (Upload)**：選擇您的對話錄音檔 (`.wav`/`.mp3`) 以及可選的註冊樣本 (例如 `Doctor_001.wav`)。
*   **檢視與編輯 (View & Edit)**：
    *   使用互動式波形圖進行導覽。
    *   直接在轉錄視窗中修正文字。
    *   儲存草稿並產生結果。
*   **產生資料集 (Generate Dataset)**：點擊 "Generate & Download" 獲取清理後的 `llm_dataset.jsonl`。

### 2. 批次處理 (CLI)
使用 `batch_folder.py` 對整個資料夾進行批量處理。

```bash
python batch_folder.py --input ./Case/dataset --output ./uploads/
```

### 3. LLM 訓練 (LoRA)
使用產出的資料集微調 Phi-3 等模型。

**一般 CPU/GPU 設備：**
```bash
python train/train_phi3.py --data_path ./datasets/my_data.jsonl --output_dir ./outputs_lora_v1
```

**Apple Silicon (M1/M2 Mac)：**
```bash
python train/train_phi3_m2.py --device mps --torch_dtype float16 --data_path ./datasets/my_data.jsonl --output_dir ./outputs_lora_v1
```

### 4. 互動式對話 (Inference)
掛載微調後的 LoRA adapter 啟動對話介面。

```bash
python train/chat_web.py --lora_dir ./outputs_lora_v1 --force_traditional
```

## 📂 專案結構

```
.
├── flaskv5.py          # 語音轉錄與分話主網頁
├── batch_folder.py     # 資料夾批次處理腳本
├── requirements.txt    # Python 依賴清單
├── train/              # LLM 訓練與對話模組
│   ├── train_phi3.py   # Phi-3 LoRA 訓練腳本 (通用版)
│   ├── train_phi3_m2.py# Phi-3 LoRA 訓練腳本 (M2 Mac 優化版)
│   ├── chat_web.py     # 對話介面後端
│   └── templates/      # 對話介面 HTML (chat_line.html)
├── uploads/            # 處理結果輸出目錄 (音訊、JSON結果)
└── Case/               # 範例資料與樣本 (註：此目錄已設為 git 忽略)
```

## 🔍 運作原理

1.  **前處理 (Preprocessing)**：將音訊轉換為 16kHz 單聲道。
2.  **轉錄 (Transcription)**：Whisper 產生帶有時間戳記的文字片段。
3.  **分話 (Diarization)**：
    *   提取每個單詞/片段的聲紋特徵 (Voice Embeddings)。
    *   與上傳的 "Enrollment" 聲紋進行比對。
    *   基於餘弦相似度 (Cosine Similarity > 0.7) 分配說話者。
4.  **資料集產生 (Dataset Generation)**：
    *   邏輯配對 "Patient" 的問題與 "Doctor" 的建議。
    *   輸出適用於微調本地 LLM (如 LLaMA, TaiwaLLM) 的 JSONL 格式。

## 📝 授權

[MIT License](LICENSE)
