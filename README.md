# 🎯 AI Resume Optimizer

An AI-powered resume optimization tool that analyzes your resume against job descriptions and rewrites it for maximum ATS (Applicant Tracking System) compatibility.

🔗 **Live App:** [resume-optimizer-kcoqjjnzomyzdygkvrztv6.streamlit.app](https://resume-optimizer-kcoqjjnzomyzdygkvrztv6.streamlit.app/)

---

## ✨ Features

- 📊 **ATS Score** — Calculates how well your resume matches a job description (0-100)
- 🔍 **Keyword Gap Analysis** — Identifies missing keywords by category
- ✍️ **AI-Powered Rewrite** — Uses Gemini AI to rewrite your resume for 75%+ ATS match
- 📈 **Before/After Comparison** — See your score improvement after AI optimization
- 📄 **PDF Export** — Download your optimized resume as a professional PDF

---

## 🚀 How to Use

1. Open the [live app](https://resume-optimizer-kcoqjjnzomyzdygkvrztv6.streamlit.app/)
2. Upload your resume as a PDF
3. Paste the job description you're applying for
4. Click **Analyze & Optimize**
5. View your ATS score, missing keywords, and AI-enhanced resume
6. Download as PDF or plain text

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web interface |
| Google Gemini AI | Resume rewriting |
| scikit-learn | TF-IDF ATS scoring |
| pypdf | PDF text extraction |
| ReportLab | PDF generation |
| Python | Backend logic |

---

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/NIKHITHA-99/Resume-Optimizer.git
cd Resume-Optimizer

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
Resume-Optimizer/
├── app.py              # Main Streamlit app (self-contained)
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignores .env and venv
└── README.md           # This file
```

---

## 🔑 API Key Setup

This app uses [Google Gemini AI](https://aistudio.google.com/):

1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Click **Get API Key** → Create API Key
3. Add it to your `.env` file locally or Streamlit Cloud secrets

---

## 👩‍💻 Built By

**Nikhitha** — [github.com/NIKHITHA-99](https://github.com/NIKHITHA-99)