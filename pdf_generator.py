"""
pdf_generator.py
Creates professional, ATS-friendly PDF resumes using reportlab.
"""

import io
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor


def create_professional_pdf(text: str) -> io.BytesIO:
    """
    Convert plain resume text into a clean, ATS-friendly PDF.
    Returns a BytesIO buffer ready for st.download_button.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ResumeTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=HexColor("#1a1d2e"),
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    contact_style = ParagraphStyle(
        "ContactInfo",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=HexColor("#555555"),
    )

    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=HexColor("#4361ee"),
        spaceAfter=4,
        spaceBefore=10,
        fontName="Helvetica-Bold",
        borderPadding=(0, 0, 2, 0),
    )

    job_title_style = ParagraphStyle(
        "JobTitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=HexColor("#1a1d2e"),
        spaceAfter=2,
        fontName="Helvetica-Bold",
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        textColor=HexColor("#2d3748"),
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=3,
        leftIndent=12,
        textColor=HexColor("#2d3748"),
    )

    story = []
    lines = text.split("\n")

    section_keywords = {
        "SUMMARY", "OBJECTIVE", "EXPERIENCE", "EDUCATION", "SKILLS",
        "PROJECTS", "CERTIFICATIONS", "ACHIEVEMENTS", "PUBLICATIONS",
        "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "TECHNICAL SKILLS",
        "PROFILE", "QUALIFICATIONS", "LEADERSHIP", "AWARDS",
    }

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.06 * inch))
            continue

        # Name — first non-empty line or short ALL CAPS
        if i < 4 and (stripped.isupper() or (len(stripped.split()) <= 5 and i == 0)):
            story.append(Paragraph(stripped, title_style))
            story.append(Spacer(1, 0.05 * inch))
            continue

        # Contact info line
        if "@" in stripped or re.search(r"\d{3}[\-.\s]?\d{3}[\-.\s]?\d{4}", stripped):
            story.append(Paragraph(stripped, contact_style))
            continue

        # Section header
        upper = stripped.upper()
        if stripped.isupper() and len(stripped) < 55:
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph(stripped, section_style))
            # Thin rule under section header
            from reportlab.platypus import HRFlowable
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=HexColor("#4361ee"), spaceAfter=4))
            continue
        if any(kw in upper for kw in section_keywords) and len(stripped) < 55:
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph(stripped.upper(), section_style))
            from reportlab.platypus import HRFlowable
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=HexColor("#4361ee"), spaceAfter=4))
            continue

        # Bullet point
        if stripped.startswith(("•", "-", "*")):
            clean = stripped.lstrip("•-* ").strip()
            story.append(Paragraph(f"&bull;&nbsp; {clean}", bullet_style))
            continue

        # Job title / sub-heading (contains | or short bold-like line)
        if "|" in stripped and len(stripped) < 120:
            story.append(Paragraph(f"<b>{stripped}</b>", job_title_style))
            continue

        # Regular paragraph
        story.append(Paragraph(stripped, body_style))

    # Build with fallback
    try:
        doc.build(story)
    except Exception:
        # Fallback: plain text only
        story = []
        for line in lines:
            if line.strip():
                story.append(Paragraph(line.strip(), body_style))
                story.append(Spacer(1, 0.05 * inch))
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        doc.build(story)

    buffer.seek(0)
    return buffer


def create_simple_pdf(text: str) -> io.BytesIO:
    """Minimal fallback PDF — plain text only."""
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []
    for para in text.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["Normal"]))
            story.append(Spacer(1, 0.08 * inch))
    doc.build(story)
    buffer.seek(0)
    return buffer