"""
analyzer.py
-----------
Core NLP analysis module.

Improvements in this version:
  1. Synonym expansion — "led a team" matches "team leadership", etc.
  2. Skill proficiency weighting — "expert in Python" scores higher than "familiar with Python"
  3. Vague JD detection — thin JDs get a fallback skill-based score instead of misleading low score
  4. Robust match score — hybrid word overlap + TF-IDF + skill match bonus
"""

import re
import json
import os
from typing import List, Dict, Tuple
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# NLTK setup
# ──────────────────────────────────────────────
def download_nltk_data():
    for pkg, path in [("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

download_nltk_data()


# ──────────────────────────────────────────────
# Synonym Map — Fix 1: Semantic matching
# Words/phrases on the right are treated as
# equivalent to the key when scoring.
# ──────────────────────────────────────────────
SYNONYMS: Dict[str, List[str]] = {
    # Leadership / management
    "leadership":       ["led a team", "team lead", "leading a team", "managed a team",
                         "team management", "managed team", "supervised", "mentored"],
    "management":       ["managed", "overseeing", "oversee", "coordinated", "coordination"],
    "communication":    ["communicated", "presented", "presentation", "stakeholder"],
    "collaboration":    ["collaborated", "cross-functional", "worked with team", "teamwork"],
    "problem solving":  ["solved", "troubleshoot", "debugging", "resolved issues", "root cause"],

    # Development actions
    "developed":        ["built", "created", "implemented", "designed", "engineered",
                         "constructed", "wrote", "coded"],
    "deployed":         ["deployment", "shipped", "released", "launched", "production"],
    "optimized":        ["improved", "enhanced", "reduced latency", "performance tuning",
                         "increased efficiency", "speeded up"],
    "tested":           ["unit test", "testing", "test cases", "qa", "quality assurance",
                         "integration test", "pytest", "junit"],

    # Tech synonyms
    "rest api":         ["restful api", "restful", "api development", "web api", "http api",
                         "microservice", "microservices", "api integration"],
    "machine learning": ["ml model", "ml pipeline", "predictive model", "supervised learning",
                         "unsupervised learning", "model training", "model building"],
    "deep learning":    ["neural network", "neural networks", "cnn", "rnn", "lstm",
                         "transformer", "bert", "fine-tuning"],
    "data analysis":    ["data analytics", "data analyst", "analyzed data", "data insights",
                         "exploratory data analysis", "eda"],
    "database":         ["sql", "mysql", "postgresql", "nosql", "mongodb", "db", "rdbms"],
    "version control":  ["git", "github", "gitlab", "source control", "vcs"],
    "cloud":            ["aws", "azure", "gcp", "google cloud", "cloud platform",
                         "cloud infrastructure", "serverless"],
    "agile":            ["scrum", "sprint", "kanban", "jira", "agile methodology"],
    "ci/cd":            ["continuous integration", "continuous deployment", "pipeline",
                         "jenkins", "github actions", "devops"],
    "python":           ["pandas", "numpy", "scikit", "sklearn", "flask", "django",
                         "fastapi", "streamlit"],
    "data science":     ["data scientist", "data engineering", "feature engineering",
                         "model evaluation", "data pipeline"],
    "nlp":              ["natural language", "text processing", "sentiment analysis",
                         "named entity", "text classification", "language model"],
    "computer vision":  ["image recognition", "object detection", "opencv", "image processing"],
    "docker":           ["containerization", "container", "kubernetes", "k8s"],
    "linux":            ["unix", "bash", "shell scripting", "command line", "terminal"],
    "javascript":       ["react", "node.js", "nodejs", "vue", "angular", "frontend",
                         "typescript", "es6"],
    "sql":              ["query", "queries", "database query", "joins", "stored procedure",
                         "relational database"],
}

# Build reverse map: synonym phrase → canonical term
_SYNONYM_REVERSE: Dict[str, str] = {}
for canonical, synonyms in SYNONYMS.items():
    for syn in synonyms:
        _SYNONYM_REVERSE[syn.lower()] = canonical


def expand_text_with_synonyms(text: str) -> str:
    """
    Expand text by appending canonical synonym terms wherever
    a synonym phrase is detected.

    For example: "led a team" → "led a team leadership management"

    This makes TF-IDF and overlap scoring synonym-aware without
    requiring word embeddings.
    """
    text_lower = text.lower()
    additions = []

    for phrase, canonical in _SYNONYM_REVERSE.items():
        if phrase in text_lower:
            # Append the canonical term so it participates in scoring
            additions.append(canonical)
            additions.append(canonical)  # double weight for confirmed synonym

    if additions:
        return text + " " + " ".join(additions)
    return text


# ──────────────────────────────────────────────
# Skill proficiency signals — Fix 2
# ──────────────────────────────────────────────

# High proficiency context words — skills near these get boosted weight
_EXPERT_SIGNALS = {
    "expert", "proficient", "strong", "advanced", "extensive",
    "experienced", "years of", "yr of", "yrs of", "mastery",
    "deep knowledge", "hands-on", "production", "led", "built",
    "implemented", "architected", "designed"
}

# Low proficiency context words — skills near these get reduced weight
_BEGINNER_SIGNALS = {
    "familiar", "exposure", "basic", "beginner", "learning",
    "introductory", "some experience", "coursework", "academic"
}

# Window size (characters) around a skill mention to check for proficiency signals
_PROFICIENCY_WINDOW = 80


def get_skill_proficiency(text: str, skill: str) -> str:
    """
    Detect the proficiency level of a skill mentioned in the text.

    Looks at the surrounding words within a character window around
    the skill mention to find proficiency signals.

    Returns:
        "expert"    — strong/advanced/years experience
        "beginner"  — familiar/basic/coursework
        "moderate"  — default when no clear signal
    """
    text_lower = text.lower()
    skill_lower = skill.lower()

    pattern = r"\b" + re.escape(skill_lower) + r"\b"
    match = re.search(pattern, text_lower)
    if not match:
        return "moderate"

    start = max(0, match.start() - _PROFICIENCY_WINDOW)
    end   = min(len(text_lower), match.end() + _PROFICIENCY_WINDOW)
    context = text_lower[start:end]

    context_words = set(context.split())

    if context_words & _EXPERT_SIGNALS:
        return "expert"
    if context_words & _BEGINNER_SIGNALS:
        return "beginner"
    return "moderate"


# ──────────────────────────────────────────────
# Skills loading
# ──────────────────────────────────────────────
def load_skills(skills_path: str = None) -> List[str]:
    """Load skills from data/skills.json, sorted longest-first."""
    if skills_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        skills_path = os.path.join(base_dir, "data", "skills.json")

    with open(skills_path, "r", encoding="utf-8") as f:
        skills_data = json.load(f)

    all_skills = []
    for skill_list in skills_data.values():
        all_skills.extend(skill_list)

    return sorted(set(all_skills), key=lambda s: len(s), reverse=True)


# ──────────────────────────────────────────────
# Text Preprocessing
# ──────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Light cleaning: lowercase, remove punctuation, normalize spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_clean(text: str) -> List[str]:
    """Tokenize and remove stopwords."""
    cleaned = preprocess_text(text)
    tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t not in stop_words and len(t) > 2]


# ──────────────────────────────────────────────
# Skill Extraction
# ──────────────────────────────────────────────
_SKIP_EXACT = {"c", "r", "go"}


def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    """
    Detect skills mentioned in text using word-boundary regex.
    Single-character ambiguous skills (C, R) require programming context.
    """
    text_lower = text.lower()
    found_skills = []

    for skill in skills_list:
        skill_lower = skill.lower()

        if skill_lower in _SKIP_EXACT:
            patterns = [
                rf"\b{re.escape(skill_lower)}\s*(?:programming|language|developer|code|coding)\b",
                rf"(?:proficient in|experience with|knowledge of)\s*[:\-]?\s*(?:[a-z,\s]*?\s)?{re.escape(skill_lower)}\b",
                rf"\b{re.escape(skill_lower)}\s*[,/]\s*(?:c\+\+|python|java|javascript)",
                rf"(?:c\+\+|python|java|javascript)\s*[,/]\s*{re.escape(skill_lower)}\b",
            ]
            if any(re.search(p, text_lower) for p in patterns):
                found_skills.append(skill)
            continue

        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills))


def extract_skills_with_proficiency(
    text: str, skills_list: List[str]
) -> Dict[str, str]:
    """
    Extract skills and their proficiency levels.

    Returns:
        Dict mapping skill name → proficiency ("expert"/"moderate"/"beginner")
    """
    skills = extract_skills(text, skills_list)
    return {skill: get_skill_proficiency(text, skill) for skill in skills}


# ──────────────────────────────────────────────
# Skill Gap Analysis with proficiency awareness
# ──────────────────────────────────────────────
SKILL_ALIASES: Dict[str, List[str]] = {
    "nlp": ["natural language processing"],
    "natural language processing": ["nlp"],
    "oop": ["object oriented programming"],
    "object oriented programming": ["oop"],
    "ml": ["machine learning"],
    "machine learning": ["ml"],
    "dl": ["deep learning"],
    "deep learning": ["dl"],
    "dsa": ["data structures and algorithms", "data structures", "algorithms"],
    "data structures and algorithms": ["dsa", "data structures", "algorithms"],
    "dbms": ["database management"],
    "database management": ["dbms"],
    "js": ["javascript"],
    "javascript": ["js"],
    "ts": ["typescript"],
    "typescript": ["ts"],
    "gcp": ["google cloud"],
    "google cloud": ["gcp"],
    "aws": ["amazon web services"],
    "amazon web services": ["aws"],
    "ci/cd": ["ci cd"],
    "ci cd": ["ci/cd"],
    "rest api": ["restful api", "api"],
    "node.js": ["nodejs", "node js"],
    "react": ["reactjs", "react.js"],
}


def _expand_with_aliases(skill_set_lower: set) -> set:
    expanded = set(skill_set_lower)
    for skill in list(skill_set_lower):
        for alias in SKILL_ALIASES.get(skill, []):
            expanded.add(alias)
    return expanded


def analyze_skill_gap(
    resume_skills: List[str],
    jd_skills: List[str],
    resume_text: str = "",
) -> Dict:
    """
    Compare resume skills vs JD skills with alias awareness and proficiency.

    Returns matched, missing, extra skill lists plus proficiency info.
    """
    resume_lower = set(s.lower() for s in resume_skills)
    jd_lower     = set(s.lower() for s in jd_skills)

    resume_expanded = _expand_with_aliases(resume_lower)
    jd_expanded     = _expand_with_aliases(jd_lower)

    jd_map     = {s.lower(): s for s in jd_skills}
    resume_map = {s.lower(): s for s in resume_skills}

    matched_lower = set()
    missing_lower = set()

    for jd_skill_lower in jd_lower:
        if jd_skill_lower in resume_expanded:
            matched_lower.add(jd_skill_lower)
        else:
            missing_lower.add(jd_skill_lower)

    extra_lower = resume_lower - set(
        r for r in resume_lower if r in jd_expanded or r in jd_lower
    )

    matched = sorted([jd_map[s] for s in matched_lower if s in jd_map])
    missing = sorted([jd_map[s] for s in missing_lower if s in jd_map])
    extra   = sorted([resume_map[s] for s in extra_lower if s in resume_map])

    # Proficiency labels for matched skills (from resume context)
    proficiency = {}
    if resume_text:
        skills_list = load_skills()
        prof_map = extract_skills_with_proficiency(resume_text, skills_list)
        for skill in matched:
            proficiency[skill] = prof_map.get(skill, "moderate")

    return {
        "matched": matched,
        "missing": missing,
        "extra":   extra,
        "proficiency": proficiency,  # {skill: "expert"/"moderate"/"beginner"}
    }


# ──────────────────────────────────────────────
# Vague JD Detection — Fix 4
# ──────────────────────────────────────────────
def is_vague_jd(jd_text: str, jd_skills: List[str]) -> bool:
    """
    Detect if a job description is vague (mostly eligibility/HR content,
    not technical requirements).

    A JD is considered vague if:
      - Fewer than 3 technical skills detected
      - Less than 100 unique meaningful words
      - High ratio of HR/eligibility keywords

    Args:
        jd_text:   Raw JD text.
        jd_skills: Skills detected in the JD.

    Returns:
        True if JD is too vague for reliable scoring.
    """
    tokens = tokenize_and_clean(jd_text)
    unique_words = set(tokens)

    # HR/eligibility signal words — if these dominate, JD is not technical
    hr_signals = {
        "eligibility", "criteria", "backlog", "arrear", "cgpa", "percentage",
        "batch", "graduation", "degree", "qualification", "university", "college",
        "fulltime", "correspondence", "stipend", "salary", "package", "bond",
        "selection", "process", "registration", "apply", "application"
    }

    hr_count = len(unique_words & hr_signals)
    hr_ratio = hr_count / max(len(unique_words), 1)

    return len(jd_skills) < 3 or len(unique_words) < 80 or hr_ratio > 0.08


def calculate_skill_based_score(
    resume_skills: List[str],
    jd_skills: List[str],
    resume_text: str,
) -> float:
    """
    Fallback score for vague JDs: based purely on skill overlap
    weighted by proficiency level.

    Expert skills count 1.0, moderate 0.7, beginner 0.4.
    Score = weighted matched skills / total JD skills * 100.
    """
    if not jd_skills:
        return 0.0

    skills_list = load_skills()
    prof_map = extract_skills_with_proficiency(resume_text, skills_list)

    resume_lower   = _expand_with_aliases(set(s.lower() for s in resume_skills))
    jd_lower       = set(s.lower() for s in jd_skills)

    weight_map = {"expert": 1.0, "moderate": 0.7, "beginner": 0.4}
    weighted_matched = 0.0

    for jd_skill in jd_skills:
        if jd_skill.lower() in resume_lower:
            prof = prof_map.get(jd_skill, "moderate")
            weighted_matched += weight_map.get(prof, 0.7)

    return round(min((weighted_matched / len(jd_skills)) * 100, 100.0), 2)


# ──────────────────────────────────────────────
# Match Score — Hybrid with synonym expansion
# ──────────────────────────────────────────────
def calculate_match_score(
    resume_text: str,
    jd_text: str,
    resume_skills: List[str] = None,
    jd_skills: List[str] = None,
) -> Tuple[float, bool]:
    """
    Calculate resume-JD match score.

    Uses 3 signals:
      1. Word overlap with synonym expansion (most important)
      2. TF-IDF cosine similarity with synonym expansion
      3. Skill match bonus (weighted by proficiency)

    Also detects vague JDs and uses skill-based scoring as fallback.

    Returns:
        (score: float, is_vague_jd: bool)
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, False

    # Detect vague JD — use skill-based fallback if detected
    vague = is_vague_jd(jd_text, jd_skills or [])
    if vague and jd_skills and resume_skills:
        score = calculate_skill_based_score(resume_skills, jd_skills, resume_text)
        # Even for vague JDs, blend with a small text signal
        # to avoid pure skill-count scoring
        text_score = _text_match_score(resume_text, jd_text)
        return round((score * 0.7 + text_score * 0.3), 2), True

    return _text_match_score(resume_text, jd_text), False


def _text_match_score(resume_text: str, jd_text: str) -> float:
    """
    Core text-based match score using synonym-expanded overlap + TF-IDF.
    """
    # Expand both texts with synonyms before scoring
    expanded_resume = expand_text_with_synonyms(resume_text)
    expanded_jd     = expand_text_with_synonyms(jd_text)

    cleaned_resume = preprocess_text(expanded_resume)
    cleaned_jd     = preprocess_text(expanded_jd)

    stop_words = set(stopwords.words("english"))

    resume_tokens = set(
        t for t in cleaned_resume.split()
        if t not in stop_words and len(t) > 2
    )
    jd_tokens = set(
        t for t in cleaned_jd.split()
        if t not in stop_words and len(t) > 2
    )

    # Signal 1: Word overlap (JD coverage weighted)
    if resume_tokens and jd_tokens:
        intersection  = resume_tokens & jd_tokens
        union         = resume_tokens | jd_tokens
        jd_coverage   = len(intersection) / len(jd_tokens)
        jaccard       = len(intersection) / len(union)
        overlap_score = (jd_coverage * 0.7 + jaccard * 0.3) * 100
    else:
        overlap_score = 0.0

    # Signal 2: TF-IDF cosine similarity
    corpus = [cleaned_resume, cleaned_jd, cleaned_resume, cleaned_jd]
    try:
        vectorizer  = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True)
        tfidf_mat   = vectorizer.fit_transform(corpus)
        similarity  = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])
        tfidf_score = float(similarity[0][0]) * 100
    except Exception:
        tfidf_score = overlap_score

    final = (overlap_score * 0.6) + (tfidf_score * 0.4)
    return round(min(final, 100.0), 2)


# ──────────────────────────────────────────────
# JD Keyword Extraction
# ──────────────────────────────────────────────
def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    """
    Extract important keywords using word frequency + bigrams.
    Uses TF on single doc (IDF is meaningless for single document).
    """
    tokens = tokenize_and_clean(text)
    freq   = Counter(tokens)

    bigrams = []
    for i in range(len(tokens) - 1):
        if len(tokens[i]) > 3 and len(tokens[i + 1]) > 3:
            bigrams.append(f"{tokens[i]} {tokens[i + 1]}")

    bigram_freq = Counter(bigrams)
    keywords    = []

    for bigram, _ in bigram_freq.most_common(top_n // 2):
        keywords.append(bigram)

    bigram_words = set(w for bg in keywords for w in bg.split())
    for word, _ in freq.most_common(top_n * 2):
        if word not in bigram_words and len(word) > 3:
            keywords.append(word)
        if len(keywords) >= top_n:
            break

    return keywords[:top_n]


# ──────────────────────────────────────────────
# Full Analysis Pipeline
# ──────────────────────────────────────────────
def run_analysis(resume_text: str, jd_text: str) -> Dict:
    """
    Run the complete NLP analysis pipeline.

    Returns a dict with match score, skills, keywords, proficiency,
    and a flag indicating whether the JD was detected as vague.
    """
    if not resume_text or len(resume_text.strip()) < 50:
        raise ValueError(
            "The resume text is too short to analyze. "
            "Please ensure your resume has readable content."
        )
    if not jd_text or len(jd_text.strip()) < 30:
        raise ValueError("The job description is too short to analyze.")

    skills_list = load_skills()

    resume_skills = extract_skills(resume_text, skills_list)
    jd_skills     = extract_skills(jd_text, skills_list)

    gap = analyze_skill_gap(resume_skills, jd_skills, resume_text)

    match_score, vague_jd = calculate_match_score(
        resume_text, jd_text, resume_skills, jd_skills
    )

    keywords = extract_keywords(jd_text, top_n=20)

    return {
        "match_score":    match_score,
        "resume_skills":  resume_skills,
        "jd_skills":      jd_skills,
        "matched_skills": gap["matched"],
        "missing_skills": gap["missing"],
        "extra_skills":   gap["extra"],
        "proficiency":    gap["proficiency"],
        "jd_keywords":    keywords,
        "vague_jd":       vague_jd,
    }
