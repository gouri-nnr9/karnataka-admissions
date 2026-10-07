from pathlib import Path
import json
import re
from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).parent
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "documents.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "chunks.json"

MAX_CHARS = 1000
OVERLAP_CHARS = 150
MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_TOKENS = 400


def get_document_metadata(source):
    metadata = {
        "exam": "PGCET",
        "year": 2026,
        "document_type": "general",
        "courses": [],
    }

    if source == "Bell_Timings_PGCET_2026_english_Mtechenglish.pdf":
        metadata["document_type"] = "bell_timings"
        metadata["courses"] = ["ME", "MTECH"]

    elif source == "PGCET_SCH_ENG_14052026english.pdf":
        metadata["document_type"] = "exam_schedule"
        metadata["courses"] = ["MBA", "MCA", "ME", "MTECH"]

    elif source == "PROF_CODE_C_mca23092026renglish.pdf":
        metadata["document_type"] = "cutoff_ranks"
        metadata["courses"] = ["MCA"]

    elif source == "pgcet_EXT_08042026english.pdf":
        metadata["document_type"] = "application_extension"
        metadata["courses"] = [
            "MBA",
            "MCA",
            "ME",
            "MTECH",
            "MARCH",
        ]

    elif source == "pgcet_notification_25092026english.pdf":
        metadata["document_type"] = "admission_result_notification"
        metadata["courses"] = [
            "MBA",
            "MCA",
            "MTECH",
            "MARCH",
        ]

    elif source == "pgcet.pdf":
        metadata["document_type"] = "admission_document"

        # Do not assign document-wide courses here.
        # This PDF contains different course-specific sections.
        metadata["courses"] = []

    return metadata


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_documents():
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def count_tokens(text):
    return len(tokenizer("passage: " + text, truncation=False)["input_ids"])


def split_oversized_text(text):
    words = text.split()
    pieces = []
    current_words = []

    for word in words:
        candidate = " ".join(current_words + [word])

        if count_tokens(candidate) <= MAX_TOKENS:
            current_words.append(word)
        else:
            if current_words:
                pieces.append(" ".join(current_words))

            current_words = [word]

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def chunk_text(text):
    paragraphs = [
        paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if count_tokens(paragraph) > MAX_TOKENS:
            if current:
                chunks.append(current)
                current = ""
            pieces = split_oversized_text(paragraph)
            chunks.extend(pieces[:-1])
            if pieces:
                current = pieces[-1]
            continue

        candidate = f"{current}\n{paragraph}".strip()

        if len(candidate) <= MAX_CHARS and count_tokens(candidate) <= MAX_TOKENS:
            current = candidate
            continue

        if current:
            chunks.append(current)

            # Keep some previous context.
            overlap = current[-OVERLAP_CHARS:]
            if " " in overlap:
                overlap = overlap.split(" ", 1)[1]
            current = f"{overlap}\n{paragraph}".strip()
        else:
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_bell_timings(text):
    pattern = (
        r"(?m)^\s*[1-7]\s*\n\s*"
        r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)"
        r"\b"
    )

    matches = list(re.finditer(pattern, text))

    # If the expected structure isn't found, safely use generic chunking.
    if len(matches) != 7:
        return chunk_text(text)

    rules_start = text.find("Only Non-Programmable Calculators")

    if rules_start < matches[-1].start():
        return chunk_text(text)

    sections = [text[: matches[0].start()]]

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else rules_start

        sections.append(text[match.start() : end])

    sections.append(text[rules_start:])

    return [section.strip() for section in sections if section.strip()]


def build_chunks(documents):
    all_chunks = []

    for document in documents:
        # text_chunks = chunk_text(document["text"])
        if document["source"].startswith("Bell_Timings"):
            text_chunks = chunk_bell_timings(document["text"])
        else:
            text_chunks = chunk_text(document["text"])

        for index, text in enumerate(text_chunks, start=1):
            chunk_id = (
                f"{Path(document['source']).stem}"
                f"_page_{document['page']}"
                f"_chunk_{index}"
            )

            document_metadata = get_document_metadata(document["source"])

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": {
                        "source": document["source"],
                        "page": document["page"],
                        "extraction_method": document["extraction_method"],
                        **document_metadata,
                    },
                }
            )

    return all_chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = build_chunks(documents)

    OUTPUT_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Saved to: {OUTPUT_PATH}")
