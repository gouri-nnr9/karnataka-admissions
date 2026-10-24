
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Forward the question to our existing FastAPI backend.
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data = await response.json();

    // Preserve FastAPI's response status, including 422 and 503.
    return NextResponse.json(data, {
      status: response.status,
    });
  } catch (error) {
    console.error("FastAPI connection error:", error);

    return NextResponse.json(
      {
        detail: "Unable to connect to the admissions service.",
      },
      { status: 503 }
    );
  }
}
