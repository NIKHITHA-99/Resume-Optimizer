# 🎯 AI Resume Optimizer

An AI-powered resume optimization tool that analyzes your resume against job descriptions and rewrites it for maximum ATS (Applicant Tracking System) compatibility.

🔗 **Live App:** [resume-optimizer-kcoqjjnzomyzdygkvrztv6.streamlit.app](https://resume-optimizer-kcoqjjnzomyzdygkvrztv6.streamlit.app/)
---

## 📌 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Run Locally](#-run-locally)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [API Key Setup](#-api-key-setup)

---

## ✨ Features

- 📊 **ATS Score** — Calculates how well your resume matches a job description (0–100)
- 🔍 **Keyword Gap Analysis** — Identifies missing keywords categorized by Technical, Soft Skills, Tools
- ✍️ **AI-Powered Rewrite** — Uses Google Gemini AI to rewrite your resume for 75%+ ATS match
- 📈 **Before/After Comparison** — See your score improvement after AI optimization
- 📄 **PDF Export** — Download your optimized resume as a professional PDF
- 📝 **Plain Text Export** — Copy-paste ready for online job application forms
- 📊 **ATS Report** — Full scoring breakdown downloadable as a text report

---

## 🎬 Demo

| Upload Resume | ATS Score | AI Enhanced |
|---|---|---|
| Upload your PDF | Get scored 0-100 | AI rewrites for 75%+ |

🔗 Try it live: [Click Here]([https://resume-optimizer-kcoqjjnzomyzdygkvrztv6.streamlit.app/)/)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core backend logic |
| **Streamlit** | Web interface & deployment |
| **Google Gemini AI** | AI-powered resume rewriting |
| **scikit-learn** | TF-IDF cosine similarity ATS scoring |
| **pypdf** | PDF text extraction |
| **ReportLab** | Professional PDF generation |
| **python-dotenv** | Environment variable management |

---

## ⚙️ How It Works

```
1. Upload Resume PDF
        ↓
2. Extract text using pypdf
        ↓
3. Compute ATS Score (TF-IDF + Keyword Match + Frequency)
        ↓
4. Identify missing keywords from Job Description
        ↓
5. Send to Google Gemini AI for rewriting
        ↓
6. Compute new ATS Score on enhanced resume
        ↓
7. Display Before/After comparison + Download options
```

**ATS Scoring Formula:**
```
Final Score = (Keyword Match x 50%) + (TF-IDF Similarity x 30%) + (Keyword Frequency x 20%)
```

---

## 💻 Run Locally

### Prerequisites
- Python 3.10+
- Google Gemini API Key (free at [aistudio.google.com](https://aistudio.google.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/NIKHITHA-99/Resume-Optimizer.git
cd Resume-Optimizer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
# Create a .env file and add:
GEMINI_API_KEY=your_api_key_here

# 5. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🚀 Deployment

This app is deployed on **Streamlit Community Cloud** (free tier).

### Deploy Your Own Copy

1. **Fork this repository** on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"**
5. Select your forked repo → branch: `main` → file: `app.py`
6. Click **"Deploy"**
7. Go to **Settings → Secrets** and add:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
8. Your app will be live at `https://your-app-name.streamlit.app` 🎉

### Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API Key | Yes |

---

## 📁 Project Structure

```
Resume-Optimizer/
├── app.py                  # Main Streamlit app (fully self-contained)
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores .env, venv/, __pycache__/
├── setup.bat               # Windows auto-installer script
└── README.md               # Project documentation
```

---

## 🔑 API Key Setup

### Get a Free Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API Key"**
4. Copy the key (starts with `AIza...`)

### Add to Local `.env` file
```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXX
```

### Add to Streamlit Cloud
```
Settings → Secrets → paste:
GEMINI_API_KEY = "AIzaSyXXXXXXXXXXXXXXXX"
```

> Never commit your `.env` file to GitHub! It is already in `.gitignore`.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a new branch
```bash
git checkout -b feature/your-feature-name
```
3. **Commit** your changes
```bash
git commit -m "Add: your feature description"
```
4. **Push** and open a **Pull Request**

### Ideas for Contributions
- Support for DOCX resume uploads
- Multi-language support
- More detailed scoring breakdown
- Support for other AI models (OpenAI, Claude)


## 👩‍💻 Built By

**Nikhitha** — [github.com/NIKHITHA-99](https://github.com/NIKHITHA-99)
