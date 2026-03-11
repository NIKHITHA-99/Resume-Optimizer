"""
utils.py
Utility functions: text cleaning, chunking, keyword extraction,
section detection, and PDF generation helper.
"""

import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


# ─── Text Cleaning ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove extra whitespace and unwanted special characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,+/#@\-() ]", "", text)
    return text.strip()


# ─── Text Chunking ───────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Split text into overlapping word-based chunks for RAG retrieval.
    Returns a list of strings.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─── Keyword Extraction ──────────────────────────────────────────
def extract_keywords(text: str) -> list:
    """Basic keyword extraction using stopword removal."""
    if not text:
        return []

    stopwords = {
        "and","or","the","a","an","to","for","of","in","on",
        "with","at","by","from","is","are","was","were","be",
        "been","have","has","had","will","would","should","can",
        "this","that","it","its","we","they","you","your","our",
    }
    words = text.lower().split()
    keywords = [
        w for w in words
        if w not in stopwords and len(w) > 2 and re.match(r"[a-z]", w)
    ]
    # Deduplicate while preserving order
    seen, result = set(), []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


# ─── Section Detection ───────────────────────────────────────────
def detect_sections(resume_text: str):
    """
    Detect which standard resume sections are present or missing.
    Returns (found: list[str], missing: list[str]).
    """
    if not resume_text:
        return [], []

    sections = {
        "education":       "Education",
        "experience":      "Experience",
        "projects":        "Projects",
        "skills":          "Skills",
        "certifications":  "Certifications",
        "achievements":    "Achievements",
        "summary":         "Summary",
        "objective":       "Objective",
    }

    text_lower = resume_text.lower()
    found   = [label for key, label in sections.items() if key in text_lower]
    missing = [label for key, label in sections.items() if key not in text_lower]
    return found, missing


# ─── PDF Generator (simple) ─────────────────────────────────────
def generate_pdf(text: str, output_path: str = "enhanced_resume.pdf") -> str:
    """
    Convert plain text resume to a simple PDF file.
    Returns the output path.
    """
    if not text:
        raise ValueError("No text provided for PDF generation.")

    styles  = getSampleStyleSheet()
    content = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            content.append(Paragraph(stripped, styles["BodyText"]))
            content.append(Spacer(1, 6))

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    doc.build(content)
    return output_path


# ─── Self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Education: BTech in AI & ML
    Experience: 2 years at TechCorp
    Skills: Python, Machine Learning, NLP
    Projects: Resume Analyzer, Chatbot
    Summary: Motivated engineer seeking ML roles.
    """

    print("Cleaned:\n", clean_text(sample))
    print("\nChunks:", len(chunk_text(sample)))
    print("\nKeywords:", extract_keywords(sample)[:10])
    found, missing = detect_sections(sample)
    print("\nFound:", found)
    print("Missing:", missing)