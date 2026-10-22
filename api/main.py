from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from google import genai

from ask import process_question, api_key

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Karnataka Admissions RAG API",
    version="1.0.0",
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty or whitespace only")

        return value


class SourceResponse(BaseModel):
    label: str
    source_file: str
    page: int


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    status: str
    sources: list[SourceResponse]


# --------------------------------------------------
# Health check
# --------------------------------------------------


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Karnataka Admissions RAG",
    }


# --------------------------------------------------
# RAG question-answering endpoint
# --------------------------------------------------


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    try:
        # Reuse our existing PostgreSQL-backed RAG pipeline.
        with genai.Client(api_key=api_key) as client:
            result = process_question(
                request.question.strip(),
                client,
            )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The question-answering service is temporarily unavailable.",
        )

    # Distinguish external API failures from valid refusals.
    if result.get("status") in ("api_error", "error"):
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable.",
        )

    # Return a structured response for our Angular frontend.
    return {
        "question": request.question,
        "answer": result.get("answer", ""),
        "answerable": result.get("answerable", False),
        "status": result.get("status"),
        "sources": [
            {
                "label": source["label"],
                "source_file": source["source_file"],
                "page": source["page"],
            }
            for source in result.get("sources", [])
        ],
    }
