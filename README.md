# DouSense — Multimodal Document Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-8b5cf6?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-14b8a6?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-a78bfa?style=for-the-badge)
![CLIP](https://img.shields.io/badge/CLIP-Embeddings-f59e0b?style=for-the-badge)

**Ask questions about your documents — text, images, and everything in between.**

</div>

---

## 1. Overview

DouSense is a **Multimodal Retrieval-Augmented Generation (RAG)** system that understands both text and images inside your documents. Upload a PDF, a plain text file, or a standalone image — then ask anything about it in natural language.

Under the hood, DouSense uses **OpenAI CLIP** to embed text and images into the same vector space, stores them in **ChromaDB**, and retrieves the most semantically relevant chunks to feed into a **Vision-Language Model (GPT-4o)** that generates grounded, accurate answers.

---

## 2. Features

| Feature | Description |
|---|---|
| 🧠 **Multimodal Embeddings** | CLIP encodes both text chunks and images into a unified vector space |
| ⚡ **Semantic Retrieval** | ChromaDB cosine-similarity search finds relevant content by meaning, not keywords |
| 💬 **Vision-Language Answers** | GPT-4o reads retrieved text and images together to generate grounded responses |
| 📝 **Text Mode** | Index and query plain `.txt` files |
| 🖼️ **Image Mode** | Upload a standalone image and ask questions about it |
| ✦ **Both Mode** | Full PDF indexing — extracts and embeds text chunks and embedded images |
| 🕓 **Query History** | Tracks recent queries across sessions |
| 🎨 **Dark UI** | Professional Orbitron-fonted dark interface built with Streamlit |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DouSense App                         │
│                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────┐   │
│  │  Upload  │───▶│ PDF/TXT/IMG │───▶│   CLIPEmbedder   │   │
│  │ (Sidebar)│    │  Processor  │    │  (text + images)  │   │
│  └──────────┘    └─────────────┘    └────────┬──────────┘   │
│                                              │              │
│                                              ▼              │
│                                    ┌─────────────────┐      │
│                                    │   ChromaDB      │      │
│                                    │  Vector Store   │      │
│                                    └────────┬────────┘      │
│                                             │               │
│  ┌──────────┐    ┌──────────────────────────▼────────────┐  │
│  │  Query   │──▶│        MultimodalRetriever            │  │
│  │  Input   │    │  (embed query → top-K similarity)     │  │
│  └──────────┘    └──────────────────────────┬────────────┘  │
│                                             │               │
│                                             ▼               │
│                                    ┌─────────────────┐      │
│                                    │  GPT-4o (LLM)   │      │
│                                    │  text + images  │      │
│                                    └────────┬────────┘      │
│                                             │               │
│                                    ┌────────▼────────┐      │
│                                    │    Answer UI    │      │
│                                    └─────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Input Modes

| Mode | Upload | What Gets Indexed |
|---|---|---|
| 📝 **Text** | `.txt` file | Text chunks split and embedded with CLIP text encoder |
| 🖼️ **Image** | `.png / .jpg / .webp` | Image embedded with CLIP vision encoder |
| ✦ **Both** | `.pdf` | Full PDF — text chunks + all images extracted and embedded |

---

## 5. Installation

### Prerequisites

- Python 3.10 or higher
- An OpenRouter or OpenAI API key
- ~700 MB disk space for the CLIP model

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/dousense.git
cd dousense
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CHROMA_PERSIST_DIR=./chroma_db
LLM_MAX_TOKENS=1024
TOP_K=5
```

**5. Run the app**
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**
