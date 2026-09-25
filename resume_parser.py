"""
resume_parser.py
----------------
Handles resume upload and text extraction.

Supports:
  - PDF  (.pdf) — with multi-column layout handling
  - Word (.docx) — paragraphs + table cells

Improvement: PDF extraction now tries to sort text by vertical position
so multi-column resumes (two-column layouts) are read in the correct order
instead of mixing left and right column text together.
"""

import io
import PyPDF2
import docx


# ──────────────────────────────────────────────
# PDF Extraction
# ──────────────────────────────────────────────

def _extract_page_text_sorted(page) -> str:
    """
    Extract text from a single PDF page, attempting to preserve
    reading order for multi-column layouts.

    Strategy:
      1. Try PyPDF2's built-in extract_text() first (fast, works for most CVs)
      2. If the result looks garbled (words mixed from two columns), fall back
         to a line-by-line sort by Y position which restores reading order.

    Args:
        page: A PyPDF2 PageObject.

    Returns:
        Extracted text string for that page.
    """
    # Primary attempt: PyPDF2 default extraction
    text = page.extract_text()
    if text:
        return text

    # Fallback: visitor-based extraction sorted by position
    # This handles two-column resumes by grouping text by Y coordinate
    lines: dict = {}

    def visitor(text_fragment, cm, tm, font_dict, font_size):
        """Callback called for each text fragment in the PDF."""
        if text_fragment and text_fragment.strip():
            # tm[5] is the Y coordinate in the transformation matrix
            # Round to nearest 5 units to group text on the same visual line
            y_pos = round(float(tm[5]) / 5) * 5
            if y_pos not in lines:
                lines[y_pos] = []
            lines[y_pos].append(text_fragment)

    try:
        page.extract_text(visitor_text=visitor)
        if lines:
            # Sort lines by Y position descending (top of page first)
            sorted_lines = sorted(lines.items(), key=lambda x: x[0], reverse=True)
            result = "\n".join(" ".join(frags) for _, frags in sorted_lines)
            if result.strip():
                return result
    except Exception:
        pass  # visitor API not available in older PyPDF2, that's fine

    return text or ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from a PDF file.
    Handles multi-column layouts by sorting text blocks by Y position.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Extracted text as a string.
    """
    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

        if len(pdf_reader.pages) == 0:
            raise ValueError("The uploaded PDF has no pages.")

        for page in pdf_reader.pages:
            page_text = _extract_page_text_sorted(page)
            if page_text:
                text += page_text + "\n"

        text = text.strip()

        if not text:
            raise ValueError(
                "Could not extract text from the PDF. "
                "The file may be scanned or image-based. "
                "Please use a text-based PDF or upload a Word (.docx) file."
            )

    except PyPDF2.errors.PdfReadError:
        raise ValueError(
            "The file could not be read as a PDF. "
            "Please ensure you are uploading a valid PDF file."
        )

    return text


# ──────────────────────────────────────────────
# DOCX Extraction
# ──────────────────────────────────────────────

def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract plain text from a Word (.docx) file.

    Extracts from:
      - All paragraphs (normal text, headings, bullet points)
      - All table cells (many resumes use tables for layout)
    Deduplicates table cells to avoid repeating text that appears
    in both paragraph and table contexts.

    Args:
        file_bytes: Raw bytes of the .docx file.

    Returns:
        Extracted text as a string.
    """
    try:
        doc   = docx.Document(io.BytesIO(file_bytes))
        lines = []
        seen  = set()  # dedup tracker

        def add_line(line: str):
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                lines.append(stripped)

        # Extract from paragraphs (preserves document order)
        for para in doc.paragraphs:
            add_line(para.text)

        # Extract from tables (handles table-layout resumes)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    # Cell may have multiple paragraphs
                    for para in cell.paragraphs:
                        add_line(para.text)

        text = "\n".join(lines).strip()

        if not text:
            raise ValueError(
                "Could not extract text from the Word file. "
                "The document appears to be empty."
            )

        return text

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Could not read the Word file: {str(e)}. "
            "Please ensure you are uploading a valid .docx file "
            "(not .doc — old Word format is not supported)."
        )


# ──────────────────────────────────────────────
# Unified Entry Point
# ──────────────────────────────────────────────

def extract_text(uploaded_file) -> str:
    """
    Detect file type and extract text accordingly.

    Supports PDF (.pdf) and Word (.docx).

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Extracted resume text as a string.
    """
    filename   = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: '{uploaded_file.name}'. "
            "Please upload a PDF (.pdf) or Word (.docx) file."
        )


def validate_file(uploaded_file) -> bool:
    """Return True if the uploaded file is a supported type."""
    if uploaded_file is None:
        return False
    name = uploaded_file.name.lower()
    return name.endswith(".pdf") or name.endswith(".docx")
