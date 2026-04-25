<div align="center">

# 🫁 MediScan AI

### Chest X-Ray Disease Detection with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-00A67E?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mediscan-ai-jzmgxsbzhstdazbwtsmch6.streamlit.app/)

<br>

> **Educational AI tool** for chest X-ray pneumonia detection using DenseNet-121, Grad-CAM explainability, and a RAG-powered clinical assistant grounded in WHO guidelines.

### 🚀 [Try the Live Demo](https://mediscan-ai-jzmgxsbzhstdazbwtsmch6.streamlit.app/)

</div>

---

## ⚠️ Medical Disclaimer

> This tool is for **educational and research purposes only**. It is **NOT** a substitute for professional clinical diagnosis. Always consult a qualified physician before making any medical decisions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **X-Ray Classification** | Detects NORMAL vs PNEUMONIA from chest X-ray images |
| 🌡️ **Grad-CAM Heatmaps** | Highlights exact lung regions that influenced the prediction |
| 💬 **Clinical RAG Assistant** | AI chatbot powered by LLaMA-3.3-70B, grounded in WHO clinical guidelines |
| 📊 **Probability Display** | Real-time confidence scores for each prediction class |
| 🎨 **Premium Dark UI** | Professional glassmorphism medical-grade interface |

---

## 🏗️ Architecture

```
MediScan AI
├── 🧠 Deep Learning Model
│   ├── Backbone:        DenseNet-121 (ImageNet pretrained)
│   ├── Fine-tuned:      DenseBlock4 + Custom Classifier Head
│   ├── Parameters:      ~2.4M trainable
│   └── Explainability:  Grad-CAM (Gradient-weighted Class Activation Maps)
│
├── 💬 Clinical RAG Assistant
│   ├── Embeddings:      sentence-transformers/all-MiniLM-L6-v2
│   ├── Vector Store:    FAISS (in-memory)
│   ├── LLM:             LLaMA-3.3-70B Versatile via Groq API
│   └── Knowledge Base:  WHO Clinical Guidelines on Pneumonia
│
└── 🖥️ Frontend
    ├── Framework:       Streamlit
    ├── Theme:           Custom Premium Dark (glassmorphism)
    └── Fonts:           Syne · Inter · Space Mono · JetBrains Mono
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Test AUC-ROC** | 0.96+ |
| **Test Accuracy** | 90.2% |
| **Training Dataset** | 5,863 chest X-rays (NIH / Kaggle) |
| **Architecture** | DenseNet-121 (121-layer CNN) |
| **Training Epochs** | 8 |
| **Optimizer** | Adam + Cosine Annealing LR |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.6 |
| Model Architecture | DenseNet-121 (torchvision) |
| Explainability | Grad-CAM |
| RAG Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | Groq API · LLaMA-3.3-70B Versatile |
| Frontend | Streamlit 1.35 |
| Data Augmentation | RandomHFlip, RandomRotation, ColorJitter |
| Class Imbalance Handling | WeightedRandomSampler |

---

## 📁 Project Structure

```
mediscan-ai/
├── app.py                  ← Main Streamlit application
├── kaggle_train.py         ← Model training script (run on Kaggle GPU)
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System dependencies
├── runtime.txt             ← Python version specification
├── README.md               ← You are here
├── .gitignore              ← Git ignore rules
├── samples/                ← Sample chest X-ray images for testing
│   ├── NORMAL_1.jpeg
│   ├── NORMAL_2.jpeg
│   ├── PNEUMONIA_1.jpeg
│   └── PNEUMONIA_2.jpeg
└── mediscan_best.pth       ← Trained model weights (NOT in git — train on Kaggle)
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/sathvik-BR/mediscan-ai.git
cd mediscan-ai
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your Groq API Key
Go to [console.groq.com](https://console.groq.com) → Sign up free → Create API Key

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_key_here"

# Mac/Linux
export GROQ_API_KEY="your_groq_key_here"
```

### 5. Add the trained model
The model file `mediscan_best.pth` is not included due to file size. Train it for free on Kaggle:

1. Go to [Kaggle](https://kaggle.com) → Create Notebook
2. Add dataset: [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
3. Enable **GPU T4** in Settings → Paste and run `kaggle_train.py`
4. Download `mediscan_best.pth` from the Output panel
5. Place it in the project root folder

### 6. Run the app
```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) 🎉

---

## 🧪 Sample X-Rays for Testing

Sample chest X-rays are available in the `/samples` folder. Upload them directly to test the app:

- `NORMAL_1.jpeg` / `NORMAL_2.jpeg` → Expected: **NORMAL**
- `PNEUMONIA_1.jpeg` / `PNEUMONIA_2.jpeg` → Expected: **PNEUMONIA**

---

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → Connect your repo
3. Set **Main file path** to `app.py`
4. Go to **Advanced settings → Secrets** → Add `GROQ_API_KEY = "your_key"`
5. Click **Deploy** 🚀

> **Live deployment:** [mediscan-ai-jzmgxsbzhstdazbwtsmch6.streamlit.app](https://mediscan-ai-jzmgxsbzhstdazbwtsmch6.streamlit.app/)

---

## 🌐 Deploy to HuggingFace Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `mediscan-ai` → SDK: **Streamlit** → Create
3. Upload: `app.py`, `requirements.txt`, `packages.txt`, `README.md`, `mediscan_best.pth`
4. Go to **Settings → Secrets** → Add `GROQ_API_KEY` = your key
5. Space auto-builds and deploys 🚀

---

## 📖 How It Works

### 1. X-Ray Classification
**DenseNet-121** pretrained on ImageNet, fine-tuned on chest X-rays. Only DenseBlock4 and the classifier head are trained — early layers remain frozen to retain ImageNet feature knowledge. WeightedRandomSampler handles the class imbalance (3,875 pneumonia vs 1,341 normal).

### 2. Grad-CAM Explainability
After prediction, **Gradient-weighted Class Activation Mapping** generates a heatmap showing which regions influenced the decision. Red/warm = high model attention. This is critical for clinical interpretability and trust.

### 3. Clinical RAG Assistant
A **Retrieval-Augmented Generation** pipeline answers clinical questions:
1. Question embedded using `all-MiniLM-L6-v2`
2. Relevant chunks retrieved from FAISS vector store
3. Context passed to **LLaMA-3.3-70B** via Groq API
4. Response grounded strictly in WHO clinical guidelines

---

## 🗂️ Dataset

**Chest X-Ray Images (Pneumonia)** — Paul Mooney · Kaggle

| Split | Normal | Pneumonia | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,863** |

🔗 [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 👨‍💻 Built By

<div align="center">

**B R Sathvik**

AI/ML Engineering Student · AMC Engineering College, Bengaluru · Batch 2023–2027

[![GitHub](https://img.shields.io/badge/GitHub-sathvik--BR-181717?style=for-the-badge&logo=github)](https://github.com/sathvik-BR)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-B%20R%20Sathvik-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/b-r-sathvik-a9b785328)

</div>

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

Made with ❤️ for education and research · Not for clinical use

</div>
