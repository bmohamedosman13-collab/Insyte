import io
from typing import Any

import fitz  # PyMuPDF


def extract_documents(
    uploaded_files: list[Any],
) -> tuple[list[dict], list[str]]:
    """
    Extract text from a list of Streamlit UploadedFile objects.

    Returns:
        docs   — list of {filename, pages: [{page_num, text}], full_text}
        errors — list of user-facing error strings for files that failed
    """
    docs: list[dict] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
            pdf = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")

            pages: list[dict] = []
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                pages.append({"page_num": page_num, "text": text})

            pdf.close()

            full_text = "\n\n".join(p["text"] for p in pages if p["text"])

            if not full_text.strip():
                errors.append(
                    f"'{uploaded_file.name}': No extractable text found. "
                    "This may be a scanned PDF — OCR is not supported in this prototype."
                )
                continue

            docs.append(
                {
                    "filename": uploaded_file.name,
                    "pages": pages,
                    "full_text": full_text,
                }
            )

        except Exception as exc:  # noqa: BLE001
            errors.append(f"'{uploaded_file.name}': Extraction failed — {exc}")

    return docs, errors
