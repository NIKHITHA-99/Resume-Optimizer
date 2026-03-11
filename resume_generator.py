"""
resume_generator.py — Fixed version
Problem: `import google.generativeai as genai` at top level crashes the whole
         module when the package isn't installed, making ALL modules fail.
Fix: lazy import inside the function + accept api_key as a parameter
     so the module always loads, and errors only appear when actually called.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_model(api_key: str = None):
    """Lazy-load Gemini. Raises a clear error if not installed."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed.\n"
            "Run: pip install google-generativeai"
        )

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "No Gemini API key found.\n"
            "Add GEMINI_API_KEY=your_key to your .env file."
        )

    genai.configure(api_key=key)

    model_priority = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    last_error = None
    for model_name in model_priority:
        try:
            model = genai.GenerativeModel(model_name)
            # Quick ping to verify model works
            model.generate_content(
                "Say OK",
                generation_config=genai.types.GenerationConfig(max_output_tokens=5),
            )
            print(f"✅ Using model: {model_name}")
            return model, genai
        except Exception as e:
            print(f"⚠️  {model_name} unavailable: {str(e)[:80]}")
            last_error = e
            continue

    raise RuntimeError(f"No Gemini model available. Last error: {last_error}")


def generate_enhanced_resume(resume_text: str, jd_text: str, api_key: str = None) -> str:
    """
    Generate an ATS-optimized resume targeting 75-95% match score.
    api_key: optional — falls back to GEMINI_API_KEY env var.
    """
    try:
        model, genai = _get_model(api_key)
    except (ImportError, ValueError, RuntimeError) as e:
        return f"[ENHANCEMENT UNAVAILABLE]\n\n{e}\n\nORIGINAL RESUME:\n{resume_text}"

    prompt = f"""
You are an elite ATS resume optimizer with 15+ years of experience.
Transform this resume to achieve a MINIMUM 75% ATS match score against the job description.

=== JOB DESCRIPTION ===
{jd_text}

=== CURRENT RESUME ===
{resume_text}

=== REQUIREMENTS ===
1. PROFESSIONAL SUMMARY — Open with a headline using JD keywords.
   Include 3-4 sentences with JD terminology. Target 5-8 keywords.

2. SKILLS SECTION — Create: Technical Skills, Core Competencies, Tools & Technologies.
   List ALL relevant JD keywords. Include variations (JavaScript/JS/ES6).
   Target 40+ relevant items.

3. EXPERIENCE — For each role:
   - Rewrite bullets mirroring JD language and action verbs
   - Add metrics: "Improved X by 40%", "Managed team of 8"
   - Inject JD keywords naturally
   - Expand each bullet to 2-3 lines

4. ATS FORMATTING RULES:
   - Standard headers: PROFESSIONAL SUMMARY, TECHNICAL SKILLS, PROFESSIONAL EXPERIENCE, EDUCATION
   - Simple bullet points (•)
   - No tables, columns, graphics, headers/footers
   - Standard dates (MM/YYYY)

5. DO NOT fabricate companies, roles, or certifications.
   DO expand existing experience with stronger language and relevant keywords.

Start immediately with the candidate's name. No preamble.
Target: 600-900 words.

Generate the enhanced resume now:
"""

    try:
        import google.generativeai as genai_mod
        response = model.generate_content(
            prompt,
            generation_config=genai_mod.types.GenerationConfig(
                temperature=0.85,
                max_output_tokens=8000,
                top_p=0.95,
                top_k=40,
            ),
        )
        enhanced = response.text.strip()

        if len(enhanced) < 100:
            return resume_text  # fallback to original

        print(f"✅ Enhanced: {len(resume_text)} → {len(enhanced)} chars")
        return enhanced

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return resume_text  # safe fallback


def verify_enhancement(original: str, enhanced: str, jd_text: str) -> str:
    """
    Compare original vs enhanced resume quality.
    Returns a plain-text quality report.
    """
    try:
        from ats_analyser import extract_all_terms, create_normalized_set
    except ImportError:
        return "ats_analyser not available for verification."

    jd_terms      = extract_all_terms(jd_text)
    orig_terms    = extract_all_terms(original)
    enh_terms     = extract_all_terms(enhanced)

    jd_norm   = create_normalized_set(jd_terms)
    orig_norm = create_normalized_set(orig_terms)
    enh_norm  = create_normalized_set(enh_terms)

    orig_matches = orig_norm & jd_norm
    enh_matches  = enh_norm  & jd_norm

    orig_pct = len(orig_matches) / max(len(jd_norm), 1) * 100
    enh_pct  = len(enh_matches)  / max(len(jd_norm), 1) * 100
    gain     = enh_pct - orig_pct

    if enh_pct >= 75:
        status = "🌟 EXCELLENT — Target achieved!"
    elif enh_pct >= 60:
        status = "✅ GOOD — Close to target"
    elif enh_pct >= 45:
        status = "⚠️ MODERATE — Add more keywords"
    else:
        status = "❌ POOR — Major revision needed"

    return (
        f"ENHANCEMENT QUALITY REPORT\n"
        f"{'='*45}\n"
        f"Original  : {len(orig_matches)}/{len(jd_norm)} keywords ({orig_pct:.1f}%)\n"
        f"Enhanced  : {len(enh_matches)}/{len(jd_norm)} keywords ({enh_pct:.1f}%)\n"
        f"Gain      : +{gain:.1f}%\n\n"
        f"Length    : {len(original)} → {len(enhanced)} chars "
        f"({len(enhanced)/max(len(original),1)*100:.0f}%)\n\n"
        f"Status    : {status}\n"
        f"{'='*45}"
    )


def extract_missing_critical_keywords(resume_text: str, jd_text: str, top_n: int = 15):
    """Return the most important missing keywords sorted by JD frequency."""
    try:
        from ats_analyser import extract_all_terms, create_normalized_set
    except ImportError:
        return []

    jd_terms     = extract_all_terms(jd_text)
    resume_terms = extract_all_terms(resume_text)

    jd_norm  = create_normalized_set(jd_terms)
    res_norm = create_normalized_set(resume_terms)
    missing  = jd_norm - res_norm

    jd_lower = jd_text.lower()
    ranked = sorted(
        missing,
        key=lambda t: jd_lower.count(t.lower()) + len(t) / 10,
        reverse=True,
    )
    return ranked[:top_n]