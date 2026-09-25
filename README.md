# 📄 AI Resume Analyzer & Job Matcher

🚀 **Live Demo:** [https://ai-resume-analyzer-jv3rtmiexkwx8uzkatk4iy.streamlit.app/](https://ai-resume-analyzer-jv3rtmiexkwx8uzkatk4iy.streamlit.app/)

A beginner-friendly, production-style portfolio project that analyzes your resume against a job description using NLP and Generative AI.

---

## 🌟 Features

- **PDF Resume Upload** — Extract text from any text-based PDF resume
- **Resume-JD Match Score** — TF-IDF cosine similarity score showing how well your resume matches the job
- **Matched Skills** — Skills present in both your resume and the JD
- **Missing Skills** — Skills required by the JD that are absent from your resume
- **Important Keywords** — Top keywords extracted from the job description
- **AI Resume Suggestions** — Google Gemini-powered improvement recommendations
- **Technical Interview Questions** — 5 AI-generated interview Q&A pairs tailored to the job

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core language |
| Streamlit | Web UI / Frontend |
| PyPDF2 | PDF text extraction |
| NLTK | Text preprocessing, tokenization |
| Scikit-learn | TF-IDF vectorization, cosine similarity |
| Pandas | Data handling |
| Google Gemini API | Generative AI suggestions |
| python-dotenv | Secure API key management |

---

## 🏗️ Architecture & Workflow

```
Upload Resume PDF
       │
       ▼
  PyPDF2 → Extract Text
       │
       ▼
  NLTK → Clean & Preprocess Text
       │
       ▼
  Skill Extraction (skills.json dictionary)
       │
       ├── Resume Skills
       └── JD Skills
              │
              ▼
       Skill Gap Analysis
       (Matched / Missing)
              │
              ▼
  TF-IDF + Cosine Similarity → Match Score
              │
              ▼
  Send to Google Gemini API
  (Resume + JD + Analysis Results)
              │
              ▼
  AI Suggestions + Interview Questions
              │
              ▼
  Display in Streamlit Dashboard
```

---

## 📁 Project Structure

```
AI-Resume-Analyzer/
│
├── app.py              # Streamlit UI — entry point
├── resume_parser.py    # PDF upload & text extraction
├── analyzer.py         # NLP analysis — TF-IDF, skills, match score
├── ai_analyzer.py      # Google Gemini API integration
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .env.example        # Template for environment variables
├── .gitignore          # Files excluded from Git
│
└── data/
    └── skills.json     # Skill dictionary (easily editable)
```

### Module Responsibilities

**`app.py`**
- Streamlit UI layout and styling
- User input handling (file upload + text area)
- Calls `resume_parser`, `analyzer`, and `ai_analyzer`
- Renders the results dashboard

**`resume_parser.py`**
- Validates uploaded file type
- Extracts text from PDF using PyPDF2
- Handles encrypted/image-based PDF errors

**`analyzer.py`**
- Text cleaning and preprocessing (NLTK)
- Skill extraction using substring + regex matching
- Skill gap analysis (matched vs. missing)
- TF-IDF vectorization + cosine similarity (Scikit-learn)
- JD keyword extraction

**`ai_analyzer.py`**
- Loads Gemini API key from `.env`
- Builds a structured prompt with analysis context
- Calls `gemini-1.5-flash` model
- Returns parsed suggestions and interview questions

**`data/skills.json`**
- Flat JSON dictionary of skills grouped by category
- Easily editable — add or remove skills as needed

---

## ⚙️ Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Steps

**1. Clone or download the project**

```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
```

**2. Create a virtual environment (recommended)**

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

**4. Set up NLTK data**

The app automatically downloads required NLTK packages on first run. If you want to do it manually:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## 🔑 Gemini API Setup

**1. Get a free API key**

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and generate a free API key.

**2. Create your `.env` file**

Copy the example file:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

**3. Add your API key**

Open `.env` and add your key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

> ⚠️ Never share your `.env` file or commit it to Git. It is already in `.gitignore`.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📖 Example Usage

1. **Upload Resume** — Click the upload area and select your PDF resume
2. **Paste Job Description** — Copy a job posting from LinkedIn, Naukri, etc. and paste it into the text area
3. **Click "Analyze Resume"** — The app will:
   - Extract text from your PDF
   - Run NLP skill extraction and TF-IDF matching
   - Call Gemini for AI-powered suggestions
4. **Review Results** — See your match score, skill gaps, keywords, and AI recommendations

---

## 📸 Screenshots

> *(Add screenshots here after running the application)*

| Section | Description |
|---|---|
| Input Panel | Resume upload + JD text area |
| Match Score | Percentage similarity score with color indicator |
| Skills Analysis | Matched (green) and Missing (red) skills |
| Keywords | Important JD keywords |
| AI Suggestions | Gemini-generated improvement tips |
| Interview Q&A | 5 tailored technical questions with hints |

---

## 🚀 Future Improvements

- [ ] Support for DOCX resume files
- [ ] Multiple job description comparison
- [ ] Resume scoring history / session storage
- [ ] Export analysis report as PDF
- [ ] More granular skill categories
- [ ] Skill proficiency level detection
- [ ] Cover letter generator using Gemini
- [ ] LinkedIn profile analysis integration

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| "Could not extract text from PDF" | Use a text-based PDF, not a scanned image |
| "GEMINI_API_KEY is not set" | Create `.env` file with your API key |
| "Invalid Gemini API key" | Check your key at aistudio.google.com |
| "Quota exceeded" | Wait for quota reset or upgrade your plan |
| NLTK download errors | Run `python -c "import nltk; nltk.download('all')"` |
| Low match score | Your resume language may differ from JD — review AI suggestions |

---

## 📚 How It Works — For Interview Preparation

### What is TF-IDF?
**TF-IDF** (Term Frequency-Inverse Document Frequency) is a technique that converts text into numbers. Words that appear frequently in your resume but are rare across documents get higher scores. This lets the algorithm measure how relevant your resume is to a specific job.

### What is Cosine Similarity?
Imagine your resume and the job description as arrows in space. **Cosine Similarity** measures the angle between them — the smaller the angle (closer to 1.0), the more similar they are. A score of 78% means your resume shares significant vocabulary and context with the job description.

### Why Gemini AI?
Google Gemini is a large language model that understands context and language nuance far beyond simple keyword matching. It reads your resume and job description together and generates human-quality advice tailored to your specific situation.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ using Python, Streamlit, and Google Gemini.*
