"""
app.py
------
Main Streamlit application — AI Resume Analyzer & Job Matcher.

Run with:
    streamlit run app.py
"""

import streamlit as st

from resume_parser import extract_text, validate_file
from analyzer import run_analysis
from ai_analyzer import get_ai_analysis

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .main-header h1 { font-size: 2.4rem; font-weight: 700; color: #1f2937; }
    .main-header p  { font-size: 1.05rem; color: #6b7280; margin-top: -0.3rem; }

    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 2rem; text-align: center;
        color: white; box-shadow: 0 4px 20px rgba(102,126,234,0.4);
    }
    .score-card .score-value { font-size: 4rem; font-weight: 800; line-height: 1; }
    .score-card .score-label { font-size: 1.1rem; opacity: 0.9; margin-top: 0.3rem; }
    .score-card .score-note  { font-size: 0.78rem; opacity: 0.7; margin-top: 0.6rem; }

    .skill-matched {
        display:inline-block; background:#d1fae5; color:#065f46;
        border-radius:20px; padding:4px 12px; margin:3px; font-size:0.88rem; font-weight:500;
    }
    .skill-missing {
        display:inline-block; background:#fee2e2; color:#991b1b;
        border-radius:20px; padding:4px 12px; margin:3px; font-size:0.88rem; font-weight:500;
    }
    .skill-extra {
        display:inline-block; background:#dbeafe; color:#1e40af;
        border-radius:20px; padding:4px 12px; margin:3px; font-size:0.88rem; font-weight:500;
    }
    .keyword-tag {
        display:inline-block; background:#f3f4f6; color:#374151;
        border:1px solid #d1d5db; border-radius:6px;
        padding:3px 10px; margin:3px; font-size:0.85rem;
    }
    .section-header {
        font-size:1.25rem; font-weight:600; color:#1f2937;
        border-left:4px solid #667eea; padding-left:10px; margin-bottom:0.8rem;
    }
    .info-note {
        background:#eff6ff; border-left:4px solid #3b82f6;
        padding:0.75rem 1rem; border-radius:0 8px 8px 0;
        font-size:0.88rem; color:#1d4ed8; margin-bottom:1rem;
    }
    hr { margin: 1.5rem 0; }
    .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session State Initialisation
# Persists results across Streamlit reruns so the
# results panel doesn't disappear when the user
# interacts with any widget.
# ──────────────────────────────────────────────
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "analyzed_filename" not in st.session_state:
    st.session_state.analyzed_filename = None

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📄 AI Resume Analyzer & Job Matcher</h1>
        <p>Analyze your resume against a job description using NLP and AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────
# Input Section
# ──────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### 📁 Upload Your Resume")
    st.caption("Supported formats: PDF (.pdf) or Word (.docx)")
    uploaded_file = st.file_uploader(
        label="Drop your resume here",
        type=["pdf", "docx"],
        help="Upload a text-based PDF or a Word .docx resume.",
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✅ Uploaded: **{uploaded_file.name}**")

with col_right:
    st.markdown("### 📋 Paste Job Description")
    st.caption("Copy the full job description from any job portal")
    job_description = st.text_area(
        label="Job Description",
        placeholder="Paste the complete job description here...",
        height=220,
        label_visibility="collapsed",
    )
    if job_description:
        st.caption(f"📝 {len(job_description.split())} words entered")

st.markdown("---")

# ──────────────────────────────────────────────
# Analyze Button
# ──────────────────────────────────────────────
_, col_btn, _ = st.columns([2, 1, 2])
with col_btn:
    analyze_clicked = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
    )

# ──────────────────────────────────────────────
# Validation & Analysis
# ──────────────────────────────────────────────
if analyze_clicked:
    errors = []

    if uploaded_file is None:
        errors.append("⚠️ Please upload a resume file (PDF or DOCX).")
    elif not validate_file(uploaded_file):
        errors.append("⚠️ Invalid file type. Only PDF (.pdf) and Word (.docx) files are accepted.")

    if not job_description or not job_description.strip():
        errors.append("⚠️ Please paste a job description before analyzing.")
    elif len(job_description.strip()) < 50:
        errors.append("⚠️ The job description is too short. Please paste the full description.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # ── File Extraction ──────────────────────
    with st.spinner("📖 Extracting text from resume..."):
        try:
            resume_text = extract_text(uploaded_file)
        except ValueError as e:
            st.error(f"❌ File Extraction Failed: {e}")
            st.stop()

    # ── NLP Analysis ────────────────────────
    with st.spinner("🧠 Running NLP analysis..."):
        try:
            analysis = run_analysis(resume_text, job_description)
        except ValueError as e:
            st.error(f"❌ Analysis Error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected analysis error: {e}")
            st.stop()

    # ── AI Analysis ─────────────────────────
    with st.spinner("✨ Generating AI suggestions via Groq..."):
        ai_result = get_ai_analysis(
            resume_text=resume_text,
            jd_text=job_description,
            matched_skills=analysis["matched_skills"],
            missing_skills=analysis["missing_skills"],
            match_score=analysis["match_score"],
        )

    # Persist results in session state so they survive reruns
    st.session_state.analysis = analysis
    st.session_state.ai_result = ai_result
    st.session_state.analyzed_filename = uploaded_file.name

# ──────────────────────────────────────────────
# Results Dashboard
# Rendered from session state — persists across reruns
# ──────────────────────────────────────────────
if st.session_state.analysis is not None:
    analysis  = st.session_state.analysis
    ai_result = st.session_state.ai_result

    st.markdown("---")
    st.markdown(
        f"## 📊 Analysis Results"
        f"<span style='font-size:0.9rem; color:#6b7280; font-weight:normal;'>"
        f" — {st.session_state.analyzed_filename}</span>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Match Score ───────────────
    score = analysis["match_score"]

    if score >= 70:
        score_context = "🟢 Strong Match"
    elif score >= 45:
        score_context = "🟡 Moderate Match"
    else:
        score_context = "🔴 Low Match — Review Suggestions Below"

    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">Resume Match Score</div>
            <div class="score-value">{score}%</div>
            <div style="font-size:1rem; margin-top:0.5rem;">{score_context}</div>
            <div class="score-note">
                ℹ️ Application-generated similarity score (word overlap + TF-IDF).
                Not an official ATS score.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Vague JD notice
    if analysis.get("vague_jd"):
        st.warning(
            "⚠️ **Vague Job Description Detected** — This JD contains mostly eligibility/HR "
            "content with few technical requirements. The match score is based on skill overlap "
            "rather than full text similarity, which is more reliable for this type of JD. "
            "For best results, use a JD that lists specific technical skills and responsibilities."
        )

    # ── Section 2: Skills ────────────────────
    st.markdown('<div class="section-header">🎯 Skill Analysis</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2, gap="medium")

    with sc1:
        matched = analysis["matched_skills"]
        proficiency = analysis.get("proficiency", {})
        st.markdown(f"**✅ Matched Skills** &nbsp; `{len(matched)} found`")
        if matched:
            tags_html = ""
            for s in matched:
                prof = proficiency.get(s, "moderate")
                if prof == "expert":
                    badge = " 🌟"
                    title = "Expert level"
                elif prof == "beginner":
                    badge = " 📖"
                    title = "Basic level"
                else:
                    badge = ""
                    title = "Moderate level"
                tags_html += f'<span class="skill-matched" title="{title}">✓ {s}{badge}</span>'
            st.markdown(tags_html, unsafe_allow_html=True)
            if proficiency:
                st.caption("🌟 Expert &nbsp; (no badge) Moderate &nbsp; 📖 Basic — based on resume context")
        else:
            st.info("No matching skills detected. Try tailoring your resume language.")

    with sc2:
        missing = analysis["missing_skills"]
        st.markdown(f"**❌ Missing Skills** &nbsp; `{len(missing)} gaps`")
        if missing:
            st.markdown(
                "".join(f'<span class="skill-missing">✗ {s}</span>' for s in missing),
                unsafe_allow_html=True,
            )
        else:
            st.success("No skill gaps found — great alignment!")

    extra = analysis["extra_skills"]
    if extra:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**ℹ️ Additional Skills in Your Resume** &nbsp; `{len(extra)} extra`")
        st.markdown(
            "".join(f'<span class="skill-extra">{s}</span>' for s in extra),
            unsafe_allow_html=True,
        )
        st.caption("These skills are in your resume but not explicitly required by this JD.")

    st.markdown("---")

    # ── Section 3: Keywords ──────────────────
    st.markdown('<div class="section-header">🔑 Important Job Keywords</div>', unsafe_allow_html=True)
    st.caption("Top keywords extracted from the job description")
    keywords = analysis["jd_keywords"]
    if keywords:
        st.markdown(
            "".join(f'<span class="keyword-tag">{kw}</span>' for kw in keywords),
            unsafe_allow_html=True,
        )
    else:
        st.info("No significant keywords extracted.")

    st.markdown("---")

    # ── Section 4: AI Suggestions ────────────
    st.markdown('<div class="section-header">🤖 AI Resume Suggestions</div>', unsafe_allow_html=True)

    if ai_result["error"]:
        st.error(f"⚠️ AI Analysis Failed: {ai_result['error']}")
        st.info(
            "💡 The NLP analysis above is still complete. "
            "To enable AI suggestions, add **GROQ_API_KEY** to your .env file. "
            "Get a free key at https://console.groq.com/keys"
        )
    else:
        st.markdown(
            '<div class="info-note">'
            "🔍 Suggestions generated by Groq AI (Llama 3.3 70B). "
            "Review all AI advice critically before applying."
            "</div>",
            unsafe_allow_html=True,
        )
        if ai_result["suggestions"]:
            st.markdown(ai_result["suggestions"])

    st.markdown("---")

    # ── Section 5: Interview Questions ───────
    st.markdown('<div class="section-header">💬 Technical Interview Questions</div>', unsafe_allow_html=True)

    if not ai_result["error"] and ai_result["interview_questions"]:
        st.caption("AI-generated questions based on this job description")
        st.markdown(ai_result["interview_questions"])
    elif not ai_result["error"]:
        st.info("Interview questions are included in the AI suggestions above.")
    else:
        st.info("Set up GROQ_API_KEY to generate tailored interview questions.")

    st.markdown("---")

    # ── Summary Stats ────────────────────────
    st.markdown('<div class="section-header">📈 Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match Score",     f"{score}%")
    c2.metric("Matched Skills",  len(analysis["matched_skills"]))
    c3.metric("Missing Skills",  len(analysis["missing_skills"]))
    c4.metric("Resume Skills",   len(analysis["resume_skills"]))

    st.success("✅ Analysis complete!")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <br><br>
    <hr style="border:0; border-top:1px solid #e5e7eb;">
    <p style="text-align:center; color:#9ca3af; font-size:0.82rem;">
        Built with Streamlit · PyPDF2 · python-docx · NLTK · Scikit-learn · Groq API (Llama 3.3 70B)
        &nbsp;|&nbsp; AI Resume Analyzer & Job Matcher
    </p>
    """,
    unsafe_allow_html=True,
)
