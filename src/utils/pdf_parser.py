"""PDF parsing utilities using pdfplumber."""

import pdfplumber


def parse_pdf(file_path: str) -> dict:
    """Extract text and tables from a PDF file, returning structured data."""
    text_pages = []
    all_tables = []

    with pdfplumber.open(file_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)

            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                headers = table[0]
                rows = table[1:]
                table_dicts = []
                for row in rows:
                    if headers:
                        table_dicts.append(dict(zip(headers, row)))
                    else:
                        table_dicts.append({str(i): v for i, v in enumerate(row)})
                all_tables.extend(table_dicts)

    full_text = "\n".join(text_pages)
    return {
        "text": full_text,
        "tables": all_tables,
        "pages": num_pages,
    }
