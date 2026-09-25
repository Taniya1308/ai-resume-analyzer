"""
ai_analyzer.py
--------------
Handles all AI interactions via Groq API (free tier).

Uses the requests library to call Groq's OpenAI-compatible REST endpoint.
No SDK or compiled extensions needed — pure HTTP.

Get a free API key at: https://console.groq.com/keys
"""

import os
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

# Groq API endpoint (OpenAI-compatible)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Models available on this Groq account — fetched live Sept 2026
# If you get 404 errors, run: python -c "import requests,os; from dotenv import load_dotenv; load_dotenv(); r=requests.get('https://api.groq.com/openai/v1/models',headers={'Authorization':f'Bearer {os.getenv(\"GROQ_API_KEY\")}'}); [print(m['id']) for m in r.json()['data']]"
GROQ_MODELS = [
    "openai/gpt-oss-120b",  # Best quality available on this account
    "openai/gpt-oss-20b",   # Faster fallback
    "qwen/qwen3.8-27b",     # Secondary fallback
]


def get_api_key() -> str:
    """
    Read the Groq API key.

    Priority:
      1. Streamlit secrets (st.secrets) — used when deployed on Streamlit Cloud
      2. Environment variable (GROQ_API_KEY) — used locally via .env file

    Raises:
        ValueError: If the key is not found in either location.
    """
    # Try Streamlit secrets first (available when deployed on Streamlit Cloud)
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key and api_key.strip():
            return api_key.strip()
    except Exception:
        pass  # st.secrets not available outside Streamlit context

    # Fall back to environment variable (local .env file)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "• Local: Add it to your .env file: GROQ_API_KEY=gsk_...\n"
            "• Deployed: Add it in Streamlit Cloud dashboard → Settings → Secrets\n"
            "Get a free key at: https://console.groq.com/keys"
        )
    return api_key.strip()


def call_groq(prompt: str, api_key: str) -> str:
    """
    Call the Groq API with a structured system + user message.

    Uses a system prompt for consistent persona and lower temperature
    for reliable structured output.

    Args:
        prompt:  The user-turn prompt.
        api_key: Groq API key (starts with gsk_).

    Returns:
        Generated text response.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a senior technical recruiter and career coach with 10+ years of experience "
        "helping software engineers and fresh graduates land jobs at top tech companies. "
        "You give honest, specific, and actionable advice. "
        "You never fabricate skills or experiences — only work with what the candidate has. "
        "You always respond in clean markdown with the exact section structure requested."
    )

    last_error = None

    for model in GROQ_MODELS:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        # Qwen 3 requires thinking mode to be explicitly disabled
        # for standard chat responses
        if "qwen" in model.lower():
            payload["thinking"] = {"type": "disabled"}

        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=90,
            )

            # Get error details for any non-200 response
            if response.status_code != 200:
                try:
                    err_msg = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text

            # 401 = bad API key — stop immediately
            if response.status_code == 401:
                raise ValueError(
                    "Invalid Groq API key (401). "
                    "Check GROQ_API_KEY in your .env file. "
                    "Keys start with 'gsk_'. Get one at: https://console.groq.com/keys"
                )

            # 429 = rate limit
            if response.status_code == 429:
                raise ValueError(
                    "Groq API rate limit reached. Please wait a minute and try again."
                )

            # 404 / 400 / 503 = model issue — try next
            if response.status_code in (400, 404, 503):
                last_error = f"Model '{model}' unavailable ({response.status_code}): {err_msg}"
                continue

            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except ValueError:
            raise

        except requests.exceptions.Timeout:
            last_error = f"Model '{model}' timed out"
            continue

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status in (400, 404, 503):
                last_error = f"Model '{model}' returned {status}"
                continue
            raise

    raise ValueError(
        f"All Groq models failed. Last error: {last_error}. "
        "Please check your API key at console.groq.com/keys."
    )


def build_prompt(
    resume_text: str,
    jd_text: str,
    matched_skills: List[str],
    missing_skills: List[str],
    match_score: float,
) -> str:
    """
    Build a detailed, well-structured prompt for the AI.

    Sends up to 4000 chars of resume and 2000 chars of JD so the
    model has enough context to give specific, meaningful advice.
    """
    matched_str = ", ".join(matched_skills) if matched_skills else "None detected"
    missing_str = ", ".join(missing_skills) if missing_skills else "None detected"

    prompt = f"""Analyze the resume below against the job description and provide career advice.

---
RESUME:
{resume_text[:4000]}

---
JOB DESCRIPTION:
{jd_text[:2000]}

---
NLP PRE-ANALYSIS (computed separately):
- Resume-JD Match Score: {match_score}%
- Skills matched: {matched_str}
- Skills missing: {missing_str}

---
Provide your response in exactly these 5 sections using markdown:

## 1. Resume Improvement Suggestions
Give 4-5 specific, actionable tips to improve this resume for this exact role.
Focus on: what to reword, what to add, what to remove, and how to better present existing experience.
Do NOT suggest fabricating experience or skills the candidate does not have.

## 2. Important Missing Keywords
List 5-6 specific keywords/phrases from the JD that are absent from the resume but the candidate
could honestly include if they have that experience. For each, explain in one sentence why it matters.

## 3. Skills to Strengthen
Based on the missing skills, list 3-4 skills the candidate should learn or deepen.
For each, give a one-line practical learning tip (e.g. specific course, project type, or resource).

## 4. Alignment Suggestions
Give 3 specific suggestions to better tailor this resume for this job:
rephrasing bullet points, reordering sections, adjusting the summary/objective, etc.

## 5. Technical Interview Questions
Generate exactly 5 technical interview questions most likely to be asked for this role.
For each, provide a 2-3 sentence answer hint that covers the key points to mention.

**Q1: [Question]**
Hint: [Answer hint]

**Q2: [Question]**
Hint: [Answer hint]

**Q3: [Question]**
Hint: [Answer hint]

**Q4: [Question]**
Hint: [Answer hint]

**Q5: [Question]**
Hint: [Answer hint]"""

    return prompt.strip()


def get_ai_analysis(
    resume_text: str,
    jd_text: str,
    matched_skills: List[str],
    missing_skills: List[str],
    match_score: float,
) -> Dict[str, str]:
    """
    Call Groq and return parsed AI analysis.

    Returns a dict with:
      - full_response: Complete raw text
      - suggestions:   Sections 1-4
      - interview_questions: Section 5
      - error: Error message or None
    """
    try:
        api_key = get_api_key()

        prompt = build_prompt(
            resume_text=resume_text,
            jd_text=jd_text,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_score=match_score,
        )

        full_text = call_groq(prompt, api_key)

        # Split into suggestions (sections 1-4) and interview questions (section 5)
        suggestions_part = full_text
        interview_part = ""

        # Try splitting on section 5 heading (various formats the model might use)
        for split_marker in ["## 5.", "## 5 ", "**Q1:", "**Q1 :"]:
            if split_marker in full_text:
                idx = full_text.index(split_marker)
                suggestions_part = full_text[:idx].strip()
                interview_part = full_text[idx:].strip()
                break

        return {
            "full_response": full_text,
            "suggestions": suggestions_part,
            "interview_questions": interview_part,
            "error": None,
        }

    except ValueError as e:
        return {"full_response": "", "suggestions": "", "interview_questions": "", "error": str(e)}

    except requests.exceptions.ConnectionError:
        return {
            "full_response": "", "suggestions": "", "interview_questions": "",
            "error": "Network error: Could not connect to Groq API. Check your internet connection.",
        }

    except requests.exceptions.Timeout:
        return {
            "full_response": "", "suggestions": "", "interview_questions": "",
            "error": "Request timed out. Please try again.",
        }

    except Exception as e:
        return {
            "full_response": "", "suggestions": "", "interview_questions": "",
            "error": f"Unexpected error: {str(e)}",
        }
