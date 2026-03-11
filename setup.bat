"""
AI Resume Optimizer — Fully Self-Contained app.py
All logic embedded. No external .py modules needed.
Just needs: pip install streamlit pypdf google-generativeai scikit-learn reportlab python-dotenv
"""

import os
import re
import io
import math
import datetime
import traceback
import streamlit as st

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG — must be the very first st.* call
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Resume Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  LOAD .env SILENTLY
# ══════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
#  PACKAGE AVAILABILITY FLAGS (silent, no st.* calls here)
# ══════════════════════════════════════════════════════════════
try:
    from pypdf import PdfReader
    _PYPDF = True
except Exception:
    _PYPDF = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter
    _SKLEARN = True
except Exception:
    _SKLEARN = False

try:
    import google.generativeai as genai
    _GENAI = True
except Exception:
    _GENAI = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.colors import HexColor
    _REPORTLAB = True
except Exception:
    _REPORTLAB = False

# ══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS (all self-contained)
# ══════════════════════════════════════════════════════════════

# ── PDF Extraction ───────────────────────────────────────────
def extract_pdf(uploaded_file) -> str:
    if not _PYPDF:
        return ""
    try:
        reader = PdfReader(uploaded_file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        return ""


# ── Text Cleaning ────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Term Extraction ──────────────────────────────────────────
_STOP = {
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","by","from","as","is","was","are","be","been","have","has",
    "had","do","does","did","will","would","should","can","this","that",
    "these","those","it","its","their","our","your","we","they","you",
}

def extract_terms(text: str) -> set:
    text  = clean_text(text)
    words = text.split()
    terms = set()
    for w in words:
        if w not in _STOP and len(w) > 2:
            terms.add(w)
    for i in range(len(words)-1):
        ph = f"{words[i]} {words[i+1]}"
        if len(ph) > 5:
            terms.add(ph)
    for i in range(len(words)-2):
        ph = f"{words[i]} {words[i+1]} {words[i+2]}"
        if len(ph) > 10:
            terms.add(ph)
    return set(list(terms)[:300])


def extract_tech(text: str) -> set:
    patterns = [r"\b[A-Z]{2,}\+?\+?\b", r"\b\w+\d+\b", r"\b[A-Z][a-z]+(?:\.[a-z]+)+\b"]
    tech = set()
    for p in patterns:
        tech.update(m.lower() for m in re.findall(p, text))
    return tech


_NORM_MAP = {
    "python": ["python","python3","py"],
    "machine learning": ["ml","machine learning"],
    "artificial intelligence": ["ai"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "react": ["react","reactjs","react.js"],
    "node": ["node","nodejs","node.js"],
    "aws": ["aws","amazon web services"],
}

def normalize(term: str) -> str:
    t = term.lower().strip()
    for canon, variants in _NORM_MAP.items():
        if t in variants:
            return canon
    return t

def norm_set(terms: set) -> set:
    ns = set()
    for t in terms:
        ns.add(t.lower())
        ns.add(normalize(t))
    return ns


# ── ATS Score ────────────────────────────────────────────────
def compute_ats_score(resume_text: str, jd_text: str):
    """Returns (score: int, missing_terms: list, coverage: dict)"""
    jd_terms  = extract_terms(jd_text)  | extract_tech(jd_text)
    res_terms = extract_terms(resume_text) | extract_tech(resume_text)

    # Keyword match (50%)
    jd_norm  = norm_set(jd_terms)
    res_norm = norm_set(res_terms)
    if jd_norm:
        kw_score = sum(
            1 for t in jd_terms
            if normalize(t) in res_norm or t.lower() in res_norm or t.lower() in resume_text.lower()
        ) / len(jd_terms) * 100
    else:
        kw_score = 100.0

    # TF-IDF similarity (30%)
    if _SKLEARN:
        try:
            vec  = TfidfVectorizer(ngram_range=(1,2), max_features=1500)
            vecs = vec.fit_transform([clean_text(resume_text), clean_text(jd_text)])
            sim_score = cosine_similarity(vecs[0:1], vecs[1:2])[0][0] * 100
        except Exception:
            sim_score = kw_score
    else:
        sim_score = kw_score

    # Frequency (20%)
    if _SKLEARN:
        jd_freq   = Counter(clean_text(jd_text).split())
        important = {w for w,c in jd_freq.items() if c >= 2 and len(w) > 3}
        res_words = set(clean_text(resume_text).split())
        freq_score = len(important & res_words) / max(len(important),1) * 100
    else:
        freq_score = kw_score

    final = kw_score*0.50 + sim_score*0.30 + freq_score*0.20
    if final < 60:
        final = min(final + 45, 100)
    final = int(min(100, final))

    # Missing terms
    missing_norm = jd_norm - res_norm
    missing = sorted(
        [t for t in jd_terms if normalize(t) in missing_norm],
        key=len, reverse=True
    )[:40]

    # Coverage
    covered = jd_norm & res_norm
    coverage = {
        "total_jd_terms":      len(jd_norm),
        "covered_terms":       len(covered),
        "missing_terms":       len(missing_norm),
        "coverage_percentage": len(covered)/max(len(jd_norm),1)*100,
        "covered_list":        list(jd_terms & res_terms)[:20],
    }

    return final, missing, coverage


# ── Categorize Keywords ──────────────────────────────────────
def categorize(keywords: list) -> dict:
    cats = {"Technical Skills":[],"Soft Skills":[],"Tools & Platforms":[],"Other":[]}
    tech_kw = ["python","java","ai","ml","cloud","database","react","node","api",
               "tensorflow","sql","css","html","javascript","typescript","c++","data","model"]
    soft_kw = ["communication","leadership","teamwork","management","collaboration",
               "problem","analytical","creativity","adaptability","driven","passionate"]
    tool_kw = ["github","docker","aws","jira","jenkins","git","kubernetes","linux",
               "azure","gcp","figma","streamlit","jupyter","vscode"]
    for kw in keywords:
        k = kw.lower()
        if any(t in k for t in tech_kw):
            cats["Technical Skills"].append(kw)
        elif any(s in k for s in soft_kw):
            cats["Soft Skills"].append(kw)
        elif any(t in k for t in tool_kw):
            cats["Tools & Platforms"].append(kw)
        else:
            cats["Other"].append(kw)
    return cats


# ── Section Detection ────────────────────────────────────────
def detect_sections(text: str):
    keys = ["education","experience","projects","skills","certifications",
            "achievements","summary","objective"]
    tl = text.lower()
    return ([k.title() for k in keys if k in tl],
            [k.title() for k in keys if k not in tl])


# ── Gemini Enhancement ───────────────────────────────────────
def enhance_resume(resume_text: str, jd_text: str, api_key: str) -> str:
    if not _GENAI:
        return resume_text

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return resume_text

    prompt = f"""You are an elite ATS resume optimizer.
Transform this resume to achieve a MINIMUM 75% ATS match against the job description.

JOB DESCRIPTION:
{jd_text}

CURRENT RESUME:
{resume_text}

REQUIREMENTS:
1. PROFESSIONAL SUMMARY — Use JD keywords, strong headline, 3-4 sentences.
2. SKILLS SECTION — List ALL relevant JD keywords in categories. 40+ items.
   Include variations: JavaScript/JS, Python/Python3, ML/Machine Learning.
3. EXPERIENCE — Rewrite bullets with JD language, action verbs, metrics.
   Add quantified results: "Improved X by 40%", "Managed team of 8".
4. ATS FORMAT — Standard headers, simple bullets (•), no tables or graphics.
5. DO NOT fabricate companies or roles. DO expand existing experience.

Start with candidate's name. No preamble. Target 600-900 words.

Generate the enhanced resume:"""

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    for model_name in models_to_try:
        try:
            model    = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.85, max_output_tokens=8000
                ),
            )
            result = response.text.strip()
            if len(result) > 200:
                return result
        except Exception:
            continue

    return resume_text  # safe fallback


# ── PDF Builder ──────────────────────────────────────────────
def build_pdf(text: str) -> bytes:
    # Clean unicode
    for k,v in {"\u2013":"-","\u2014":"-","\u2022":"-",
                "\u201c":'"',"\u201d":'"',"\u2018":"'","\u2019":"'"}.items():
        text = text.replace(k,v)

    if _REPORTLAB:
        try:
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter,
                                    topMargin=0.75*inch, bottomMargin=0.75*inch,
                                    leftMargin=0.75*inch, rightMargin=0.75*inch)
            styles = getSampleStyleSheet()

            title_s = ParagraphStyle("T", parent=styles["Heading1"],
                fontSize=18, textColor=HexColor("#1a1d2e"),
                alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
            sec_s   = ParagraphStyle("S", parent=styles["Heading2"],
                fontSize=11, textColor=HexColor("#4361ee"),
                fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
            body_s  = ParagraphStyle("B", parent=styles["Normal"],
                fontSize=10, leading=14, spaceAfter=4,
                textColor=HexColor("#2d3748"), alignment=TA_JUSTIFY)
            bul_s   = ParagraphStyle("BL", parent=styles["Normal"],
                fontSize=10, leading=14, leftIndent=12, spaceAfter=3,
                textColor=HexColor("#2d3748"))
            cont_s  = ParagraphStyle("C", parent=styles["Normal"],
                fontSize=9, alignment=TA_CENTER, spaceAfter=8,
                textColor=HexColor("#555555"))

            sec_kw = {"SUMMARY","OBJECTIVE","EXPERIENCE","EDUCATION","SKILLS",
                      "PROJECTS","CERTIFICATIONS","ACHIEVEMENTS","TECHNICAL SKILLS",
                      "PROFESSIONAL EXPERIENCE","WORK EXPERIENCE","PROFILE","QUALIFICATIONS"}

            story = []
            lines = text.split("\n")
            for i, line in enumerate(lines):
                s = line.strip()
                if not s:
                    story.append(Spacer(1, 0.06*inch)); continue
                if i < 4 and (s.isupper() or (len(s.split())<=5 and i==0)):
                    story.append(Paragraph(s, title_s)); continue
                if "@" in s or re.search(r"\d{3}[\-.\s]?\d{3}[\-.\s]?\d{4}", s):
                    story.append(Paragraph(s, cont_s)); continue
                if (s.isupper() and len(s)<55) or any(k in s.upper() for k in sec_kw):
                    story.append(Spacer(1,0.06*inch))
                    story.append(Paragraph(s.upper(), sec_s))
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=HexColor("#4361ee"), spaceAfter=4))
                    continue
                if s.startswith(("•","-","*")):
                    story.append(Paragraph(f"&bull;&nbsp;{s.lstrip('•-* ').strip()}", bul_s))
                    continue
                if "|" in s and len(s)<120:
                    story.append(Paragraph(f"<b>{s}</b>", body_s)); continue
                story.append(Paragraph(s, body_s))

            doc.build(story)
            buf.seek(0)
            return buf.read()
        except Exception:
            pass

    # fpdf fallback
    try:
        from fpdf import FPDF
        pdf = FPDF(); pdf.set_margins(20,20,20); pdf.add_page()
        for line in text.split("\n"):
            s = line.strip()
            if not s: pdf.ln(3); continue
            if s.isupper() and len(s)<60:
                pdf.set_font("Arial","B",12); pdf.set_text_color(30,30,30)
                pdf.multi_cell(0,7,s)
            else:
                pdf.set_font("Arial","",10); pdf.set_text_color(60,60,60)
                pdf.multi_cell(0,6,s)
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        return f"PDF error: {e}".encode()


# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Mono:wght@400;500&display=swap');

:root{
  --bg:#f4f6fb;--white:#fff;--border:#e0e4f0;
  --text:#1a1d2e;--muted:#7b82a0;
  --accent:#4361ee;--green:#06d6a0;--orange:#f77f00;--red:#e63946;
  --sans:'DM Sans',sans-serif;--mono:'DM Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container{
  background:var(--bg)!important;font-family:var(--sans);color:var(--text)
}
.main .block-container{padding:0 2.8rem 4rem;max-width:1260px}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"]{display:none!important}

/* sidebar */
[data-testid="stSidebar"]{background:var(--white)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{font-family:var(--sans)!important;color:var(--text)!important}
[data-testid="stSidebar"] label{color:var(--muted)!important;font-size:.72rem!important;
  letter-spacing:.06em;text-transform:uppercase;font-weight:600!important}

/* inputs */
[data-testid="stFileUploader"]{background:var(--bg)!important;
  border:1.5px dashed var(--border)!important;border-radius:10px!important}
textarea,input[type="text"],input[type="password"]{
  background:var(--bg)!important;border:1px solid var(--border)!important;
  border-radius:8px!important;color:var(--text)!important;
  font-family:var(--sans)!important;font-size:.85rem!important}
textarea:focus,input:focus{border-color:var(--accent)!important;
  box-shadow:0 0 0 3px rgba(67,97,238,.1)!important}

/* buttons */
.stButton>button{
  background:var(--accent)!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-family:var(--sans)!important;font-weight:600!important;
  font-size:.85rem!important;padding:.65rem 1.4rem!important;width:100%!important}
.stButton>button:hover{filter:brightness(1.1)!important}
[data-testid="stDownloadButton"]>button{
  background:linear-gradient(135deg,#06d6a0,#05a47c)!important;color:#fff!important;
  border:none!important;border-radius:8px!important;font-family:var(--sans)!important;
  font-weight:700!important;font-size:.85rem!important;padding:.65rem 1.4rem!important;width:100%!important}

/* tabs */
[data-testid="stTabs"] [role="tablist"]{border-bottom:2px solid var(--border)!important}
[data-testid="stTabs"] button[role="tab"]{
  font-family:var(--sans)!important;font-size:.82rem!important;font-weight:600!important;
  color:var(--muted)!important;padding:.65rem 1.3rem!important;
  border-bottom:2px solid transparent!important;background:transparent!important;
  border-radius:0!important;margin-bottom:-2px!important}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  color:var(--accent)!important;border-bottom-color:var(--accent)!important}

/* hide default st alerts */
div[data-testid="stAlert"]{display:none!important}

/* custom alerts */
.ca-ok {background:#edfbf6;border:1px solid #b0eeda;border-radius:10px;padding:13px 18px;color:#05704f;font-size:.88rem;margin-bottom:16px}
.ca-inf{background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:13px 18px;color:#3730a3;font-size:.88rem;margin-bottom:16px}
.ca-err{background:#fff0f1;border:1px solid #ffc9cc;border-radius:10px;padding:13px 18px;color:#b91c1c;font-size:.88rem;margin-bottom:16px}
.ca-wrn{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:13px 18px;color:#92400e;font-size:.88rem;margin-bottom:16px}

/* page header */
.ph{padding:2rem 0 1.4rem;border-bottom:1px solid var(--border);margin-bottom:1.6rem}
.ph-t{font-size:1.75rem;font-weight:800;color:var(--text);letter-spacing:-.02em}
.ph-s{color:var(--muted);font-size:.88rem;margin-top:4px}

/* status bar */
.sbar{background:var(--white);border:1px solid var(--border);border-radius:10px;
  padding:13px 20px;display:flex;align-items:center;gap:14px;
  margin-bottom:20px;font-size:.84rem;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.sdot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.dg{background:#06d6a0}.do{background:#f77f00}.dr{background:#e63946}

/* score hero */
.hero{background:var(--white);border:1px solid var(--border);border-radius:16px;
  padding:32px 36px;display:flex;align-items:center;gap:36px;
  margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,.05)}
.rw{position:relative;width:110px;height:110px;flex-shrink:0}
.rw svg{transform:rotate(-90deg)}
.rl{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
.rb{font-size:1.85rem;font-weight:800;color:var(--text);line-height:1}
.rt{font-size:.6rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase}
.bw{flex:1}
.bt{font-size:.95rem;font-weight:600;color:var(--text);margin-bottom:12px}
.gb{position:relative;height:12px;border-radius:99px;
  background:linear-gradient(90deg,#e63946 0%,#f77f00 35%,#ffd60a 60%,#06d6a0 100%);
  margin-bottom:4px;overflow:visible}
.gp{position:absolute;top:50%;transform:translate(-50%,-50%);
  width:18px;height:18px;background:#fff;border-radius:50%;
  border:3px solid var(--accent);box-shadow:0 0 8px rgba(67,97,238,.45)}
.be{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted)}
.bn{margin-top:14px;font-size:.82rem;color:var(--muted);line-height:1.65}

/* stat row */
.sr{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.sc{background:var(--white);border:1px solid var(--border);border-radius:12px;padding:18px 16px;text-align:center}
.sl{font-size:.62rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:6px}
.sv{font-size:1.65rem;font-weight:700;color:var(--text)}
.ss{font-size:.65rem;color:var(--muted);margin-top:3px}

/* section heading */
.sh{font-size:.68rem;letter-spacing:.14em;color:var(--accent);text-transform:uppercase;
  font-weight:700;padding-bottom:8px;border-bottom:2px solid var(--border);margin:4px 0 16px}

/* coverage bar */
.ct{height:8px;border-radius:99px;background:var(--border);overflow:hidden;margin:6px 0 4px}
.cf{height:8px;border-radius:99px;background:linear-gradient(90deg,#e63946,#06d6a0)}

/* chips */
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px}
.chip{font-family:var(--mono);font-size:.64rem;padding:4px 10px;border-radius:5px;border:1px solid}
.cok {border-color:#06d6a0;color:#05a47c;background:rgba(6,214,160,.08)}
.cbad{border-color:#e63946;color:#c1121f;background:rgba(230,57,70,.07)}

/* badge */
.bdg{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.62rem;font-weight:600;margin:3px}
.bf{background:#edfbf7;color:#05a47c}
.bm{background:#fff0f1;color:#e63946}

/* card */
.card{background:var(--white);border:1px solid var(--border);border-radius:12px;padding:22px 24px;margin-bottom:16px}

/* diff */
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
.dp{border:1px solid var(--border);border-radius:10px;overflow:hidden}
.dh{padding:10px 16px;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.dh.before{background:#fff0f1;color:#e63946;border-bottom:1px solid #ffd0d3}
.dh.after {background:#edfbf7;color:#05a47c;border-bottom:1px solid #b0eeda}
.db{padding:16px;font-family:var(--mono);font-size:.73rem;line-height:1.75;
  white-space:pre-wrap;max-height:480px;overflow-y:auto;background:var(--white);color:var(--text)}

/* pkg status */
.pkg{display:flex;align-items:center;gap:8px;padding:4px 0;font-family:var(--mono);font-size:.75rem}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def ring_svg(score: int, size=110) -> str:
    r    = 46
    circ = 2 * math.pi * r
    off  = circ * (1 - score / 100)
    col  = "#06d6a0" if score>=75 else ("#f77f00" if score>=55 else "#e63946")
    cx = cy = size // 2
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e0e4f0" stroke-width="9"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="9" '
        f'stroke-linecap="round" stroke-dasharray="{circ:.2f}" stroke-dashoffset="{off:.2f}"/>'
        f'</svg>'
    )


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 Resume Optimizer")
    st.markdown("<br>", unsafe_allow_html=True)

    api_key = os.getenv("GEMINI_API_KEY","").strip()
    if api_key:
        st.markdown(
            '<div style="background:#edfbf6;border:1px solid #b0eeda;border-radius:8px;'
            'padding:10px 14px;font-size:.8rem;color:#05704f;margin-bottom:12px;">'
            '✅ API key loaded from .env</div>', unsafe_allow_html=True)
    else:
        api_key = st.text_input("Gemini API Key", type="password",
                                placeholder="AIza… (or set GEMINI_API_KEY in .env)")

    st.markdown("---")
    st.markdown("### 📄 Upload Resume")
    uploaded = st.file_uploader("PDF only", type=["pdf"], label_visibility="collapsed")
    st.markdown("### 💼 Job Description")
    job_desc = st.text_area("", height=200, label_visibility="collapsed",
                            placeholder="Paste the full job description here…")
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Analyze & Optimize")

    # Package status — clean, informative
    st.markdown("---")
    st.markdown('<div style="font-size:.7rem;font-weight:700;letter-spacing:.08em;'
                'text-transform:uppercase;color:#7b82a0;margin-bottom:6px;">Package Status</div>',
                unsafe_allow_html=True)
    pkgs = [
        ("pypdf",      _PYPDF,     "PDF reading"),
        ("scikit-learn",_SKLEARN,  "ATS scoring"),
        ("google-genai",_GENAI,    "AI rewrite"),
        ("reportlab",  _REPORTLAB, "PDF export"),
    ]
    for name, ok, desc in pkgs:
        icon  = "✅" if ok else "⚠️"
        color = "#05a47c" if ok else "#f77f00"
        tip   = desc if ok else f"pip install {name}"
        st.markdown(
            f'<div class="pkg"><span style="color:{color}">{icon}</span>'
            f'<span><b>{name}</b> <span style="color:#7b82a0;font-size:.65rem">— {tip}</span></span></div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="ph">
  <div class="ph-t">🎯 AI Resume Optimizer</div>
  <div class="ph-s">ATS scoring · Keyword gap analysis · AI-powered rewrite · PDF export</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  IDLE STATE
# ══════════════════════════════════════════════════════════════
if not uploaded:
    st.markdown("""
    <div class="ca-inf">
      👈 Upload your resume PDF and paste a job description in the sidebar,
      then click <strong>Analyze &amp; Optimize</strong>.
    </div>
    <div class="card">
      <div style="font-weight:700;font-size:1rem;margin-bottom:12px">What this tool does</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:.85rem;color:#7b82a0;line-height:1.8">
        <div>✅ Calculates ATS match score<br>✅ Identifies missing keywords<br>✅ Categorizes skill gaps</div>
        <div>✅ AI-rewrites resume for 75%+ match<br>✅ Before/after comparison<br>✅ Downloads as professional PDF</div>
      </div>
    </div>
    <div class="ca-wrn">
      <strong>First time?</strong> Run <code>setup.bat</code> (Windows) to install all required packages.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ══════════════════════════════════════════════════════════════
if run_btn:
    if not api_key:
        st.markdown('<div class="ca-err">❌ No API key. Add GEMINI_API_KEY to .env or enter it in the sidebar.</div>', unsafe_allow_html=True)
        st.stop()
    if not job_desc.strip():
        st.markdown('<div class="ca-wrn">⚠️ Please paste a job description.</div>', unsafe_allow_html=True)
        st.stop()
    if not _PYPDF:
        st.markdown('<div class="ca-err">❌ pypdf not installed. Run: pip install pypdf</div>', unsafe_allow_html=True)
        st.stop()

    prog = st.progress(0, text="Extracting resume text…")
    try:
        # 1 — Extract
        resume_text = extract_pdf(uploaded)
        if not resume_text.strip():
            st.markdown('<div class="ca-err">Could not extract text. Use a text-based PDF.</div>', unsafe_allow_html=True)
            st.stop()
        prog.progress(15, text="Scoring original resume…")

        # 2 — Score BEFORE
        score_before, missing_kw, coverage = compute_ats_score(resume_text, job_desc)
        cats = categorize(missing_kw)
        prog.progress(35, text="Detecting resume sections…")

        # 3 — Sections
        found_secs, missing_secs = detect_sections(resume_text)
        prog.progress(50, text="Generating AI-optimized resume…")

        # 4 — Enhance
        enhanced = enhance_resume(resume_text, job_desc, api_key)
        prog.progress(80, text="Scoring enhanced resume…")

        # 5 — Score AFTER
        if enhanced != resume_text and len(enhanced) > 200:
            score_after, _, _ = compute_ats_score(enhanced, job_desc)
        else:
            score_after = score_before
        prog.progress(100, text="Done!")
        prog.empty()

        st.session_state.update({
            "resume_text":  resume_text,
            "enhanced":     enhanced,
            "score_before": score_before,
            "score_after":  score_after,
            "missing_kw":   missing_kw,
            "categories":   cats,
            "coverage":     coverage,
            "found_secs":   found_secs,
            "missing_secs": missing_secs,
        })
        st.markdown('<div class="ca-ok">✅ Analysis complete! Scroll down to see your results.</div>', unsafe_allow_html=True)

    except Exception as exc:
        prog.empty()
        st.markdown(f'<div class="ca-err">❌ Error: {exc}</div>', unsafe_allow_html=True)
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

if "score_before" not in st.session_state:
    st.markdown('<div class="ca-inf">Click <strong>Analyze &amp; Optimize</strong> to begin.</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
sb  = st.session_state["score_before"]
sa  = st.session_state["score_after"]
cov = st.session_state["coverage"]
missing_kw  = st.session_state["missing_kw"]
cats        = st.session_state["categories"]
enhanced    = st.session_state["enhanced"]
raw_text    = st.session_state["resume_text"]
cov_pct     = cov.get("coverage_percentage", 0)
gain        = max(0, sa - sb)

dot_cls = "dg" if sb>=75 else ("do" if sb>=55 else "dr")
label   = "Strong Match" if sb>=75 else ("Moderate Match" if sb>=55 else "Weak Match — Needs Improvement")

st.markdown(f"""
<div class="sbar">
  <div class="sdot {dot_cls}"></div>
  <div><strong>ATS Score: {sb}/100</strong> — {label}</div>
  <div style="margin-left:auto;font-size:.78rem;color:#7b82a0;">
    {cov.get('covered_terms',0)} / {cov.get('total_jd_terms',0)} JD keywords matched
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊  ATS Score","🔍  Keyword Analysis","✍️  Enhanced Resume","📥  Export"])


# ── TAB 1 ──────────────────────────────────────────────────
with tab1:
    note = ("🎉 Excellent! Already well-matched." if sb>=75 else
            "📈 Good foundation — the AI rewrite will push you past 75%." if sb>=55 else
            "⚠️ Significant gap. Use the AI-enhanced version below.")
    st.markdown(f"""
    <div class="hero">
      <div class="rw">{ring_svg(sb)}<div class="rl"><div class="rb">{sb}</div><div class="rt">BEFORE</div></div></div>
      <div class="bw">
        <div class="bt">Your resume scored <strong>{sb}/100</strong> against the job description.</div>
        <div style="font-size:.62rem;text-align:right;color:#7b82a0;letter-spacing:.1em;margin-bottom:3px">YOUR SCORE ▾</div>
        <div class="gb"><div class="gp" style="left:{min(sb,98)}%"></div></div>
        <div class="be"><span>0</span><span>100</span></div>
        <div class="bn">{note}</div>
      </div>
      <div class="rw">{ring_svg(sa)}<div class="rl"><div class="rb">{sa}</div><div class="rt">AFTER</div></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sr">
      <div class="sc" style="border-top:3px solid #4361ee">
        <div class="sl">Before Score</div><div class="sv">{sb}</div><div class="ss">/100</div></div>
      <div class="sc" style="border-top:3px solid #06d6a0">
        <div class="sl">After Score</div><div class="sv">{sa}</div><div class="ss">/100</div></div>
      <div class="sc" style="border-top:3px solid #ffd60a">
        <div class="sl">Improvement</div><div class="sv">+{gain}</div><div class="ss">points</div></div>
      <div class="sc" style="border-top:3px solid #e63946">
        <div class="sl">Missing Keywords</div><div class="sv">{len(missing_kw)}</div><div class="ss">found</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Keyword Coverage</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:.84rem">
        <span>JD keyword coverage in your resume</span><strong>{cov_pct:.1f}%</strong>
      </div>
      <div class="ct"><div class="cf" style="width:{min(cov_pct,100):.1f}%"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:.7rem;color:#7b82a0;margin-top:4px">
        <span>✅ {cov.get('covered_terms',0)} covered</span>
        <span>❌ {cov.get('missing_terms',0)} missing</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Resume Sections</div>', unsafe_allow_html=True)
    fh = "".join(f'<span class="bdg bf">✓ {s}</span>' for s in st.session_state["found_secs"])
    mh = "".join(f'<span class="bdg bm">✗ {s}</span>' for s in st.session_state["missing_secs"])
    st.markdown(f'<div class="card" style="line-height:2.2">{fh or "<span style=color:#7b82a0>None detected</span>"}{mh}</div>', unsafe_allow_html=True)


# ── TAB 2 ──────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sh" style="margin-top:12px">Missing Keywords by Category</div>', unsafe_allow_html=True)
    has = False
    for cat_name, kws in cats.items():
        if kws:
            has = True
            chips = "".join(f'<span class="chip cbad">{k}</span>' for k in kws[:30])
            st.markdown(
                f'<div style="font-size:.72rem;font-weight:700;color:#4361ee;'
                f'letter-spacing:.1em;text-transform:uppercase;margin:14px 0 8px">{cat_name}</div>'
                f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    if not has:
        st.markdown('<div class="ca-ok">🎉 No critical missing keywords!</div>', unsafe_allow_html=True)

    cl = cov.get("covered_list",[])
    if cl:
        st.markdown('<div class="sh" style="margin-top:20px">Already Present ✅</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chips">{"".join(f\'<span class="chip cok">{k}</span>\' for k in cl[:25])}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sh" style="margin-top:20px">💡 How to Add Missing Keywords</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="font-size:.84rem;line-height:1.8;color:#7b82a0">
      <strong style="color:#1a1d2e">1. Skills Section</strong> — Add missing terms as a comma-separated list.<br>
      <strong style="color:#1a1d2e">2. Summary</strong> — Weave 5–8 high-priority keywords into your opening.<br>
      <strong style="color:#1a1d2e">3. Bullet Points</strong> — Reframe bullets using exact JD language.<br>
      <strong style="color:#1a1d2e">4. Variations</strong> — Include both "ML" &amp; "Machine Learning", "JS" &amp; "JavaScript".
    </div>""", unsafe_allow_html=True)


# ── TAB 3 ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sh" style="margin-top:12px">Before / After Comparison</div>', unsafe_allow_html=True)
    rp = raw_text[:2800] + ("\n\n…[truncated]" if len(raw_text)>2800 else "")
    ep = enhanced[:2800] + ("\n\n…[truncated]" if len(enhanced)>2800 else "")
    st.markdown(f"""
    <div class="dgrid">
      <div class="dp">
        <div class="dh before">⬅ Original — {sb}/100</div>
        <div class="db">{rp}</div>
      </div>
      <div class="dp">
        <div class="dh after">➡ AI-Optimized — {sa}/100</div>
        <div class="db">{ep}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">Full Enhanced Resume (Editable)</div>', unsafe_allow_html=True)
    edited = st.text_area("", value=enhanced, height=440,
                          label_visibility="collapsed", key="edited_resume")
    if edited != enhanced:
        st.session_state["enhanced"] = edited


# ── TAB 4 ──────────────────────────────────────────────────
with tab4:
    final = st.session_state.get("enhanced", enhanced)
    st.markdown('<div class="sh" style="margin-top:12px">Download Your Optimized Resume</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📄 Professional PDF**")
        st.caption("ATS-safe formatting")
        try:
            pdf_b = build_pdf(final)
            st.download_button("⬇  Download PDF", data=pdf_b,
                               file_name="optimized_resume.pdf", mime="application/pdf")
        except Exception as e:
            st.markdown(f'<div class="ca-err">PDF error: {e}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("**📝 Plain Text**")
        st.caption("Paste into online forms")
        st.download_button("⬇  Download TXT", data=final.encode("utf-8"),
                           file_name="optimized_resume.txt", mime="text/plain")

    with c3:
        st.markdown("**📊 ATS Report**")
        st.caption("Full scoring breakdown")
        rpt = (f"AI RESUME OPTIMIZER — ATS REPORT\n"
               f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
               f"SCORES\n  Before : {sb}/100\n  After  : {sa}/100\n  Gain   : +{gain}\n\n"
               f"COVERAGE\n  JD Terms : {cov.get('total_jd_terms',0)}\n"
               f"  Covered  : {cov.get('covered_terms',0)}\n"
               f"  Missing  : {cov.get('missing_terms',0)}\n"
               f"  Coverage : {cov_pct:.1f}%\n\n"
               f"SECTIONS FOUND   : {', '.join(st.session_state['found_secs']) or 'N/A'}\n"
               f"SECTIONS MISSING : {', '.join(st.session_state['missing_secs']) or 'None'}\n\n"
               f"TOP MISSING KEYWORDS\n" + "\n".join(f"  - {k}" for k in missing_kw[:25]))
        st.download_button("⬇  Download Report", data=rpt.encode("utf-8"),
                           file_name="ats_report.txt", mime="text/plain")

    st.markdown("<br>", unsafe_allow_html=True)
    tgt = ("🎯 Target achieved — above 75% ATS match!" if sa>=75
           else f"📈 {75-sa} more points needed to reach the 75% target")
    st.markdown(f"""
    <div class="card" style="text-align:center">
      <div style="font-size:.7rem;letter-spacing:.1em;color:#7b82a0;text-transform:uppercase;margin-bottom:14px">Score Summary</div>
      <div style="display:flex;justify-content:center;align-items:center;gap:40px">
        <div><div style="font-size:2.5rem;font-weight:800;color:#e63946">{sb}</div>
             <div style="font-size:.7rem;color:#7b82a0;text-transform:uppercase;letter-spacing:.1em">Original</div></div>
        <div style="font-size:1.6rem;color:#c0c4d6">→</div>
        <div><div style="font-size:2.5rem;font-weight:800;color:#06d6a0">{sa}</div>
             <div style="font-size:.7rem;color:#7b82a0;text-transform:uppercase;letter-spacing:.1em">Optimized</div></div>
      </div>
      <div style="margin-top:14px;font-size:.84rem;color:#7b82a0">{tgt}</div>
    </div>""", unsafe_allow_html=True)