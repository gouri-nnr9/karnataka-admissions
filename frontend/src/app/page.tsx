"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

type Source = {
  label: string;
  source_file: string;
  page: number;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  answerable?: boolean;
  isError?: boolean;
};

type AskResponse = {
  question: string;
  answer: string;
  answerable: boolean;
  status: string;
  sources: Source[];
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || loading) return;

    setMessages((previous) => [...previous, { role: "user", content: text }]);
    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });

      const data: AskResponse | { detail?: unknown } = await response.json();
      if (!response.ok) {
        throw new Error(
          response.status === 503
            ? "The admissions service is temporarily unavailable. Please try again."
            : response.status === 422
              ? "Please enter a valid question."
              : "Something went wrong while processing your question."
        );
      }

      const result = data as AskResponse;
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: result.answer,
          answerable: result.answerable,
          sources: result.answerable ? result.sources : [],
        },
      ]);
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            error instanceof Error
              ? error.message
              : "Unable to connect to the admissions service.",
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-50 p-4">
      <div className="flex h-[85vh] min-h-[500px] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <header className="border-b border-slate-200 px-6 py-5">
          <h1 className="text-xl font-bold text-slate-900">
            🎓 Karnataka Admissions Assistant
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Ask questions about PGCET admissions in English or Kannada.
          </p>
        </header>

        <section aria-live="polite" className="flex-1 space-y-5 overflow-y-auto px-5 py-6">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-4 text-4xl">🎓</div>
              <h2 className="text-lg font-semibold text-slate-900">
                How can I help you today?
              </h2>
              <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                Ask about examination dates, admission notifications, schedules,
                and information from official documents.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 ${
                  message.role === "user"
                    ? "bg-indigo-600 text-white"
                    : message.isError
                      ? "bg-rose-50 text-rose-800"
                      : "bg-slate-100 text-slate-800"
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.role === "assistant" &&
                  message.answerable &&
                  message.sources &&
                  message.sources.length > 0 && (
                    <div className="mt-4 border-t border-slate-200 pt-3">
                      <p className="mb-2 text-xs font-semibold text-slate-600">Sources supplied</p>
                      <div className="space-y-2">
                        {message.sources.map((source) => (
                          <p key={source.label} className="break-words text-xs text-slate-500">
                            [{source.label}] {source.source_file} — page {source.page}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl bg-slate-100 px-4 py-3 text-sm text-slate-500">
                Searching official documents...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </section>

        <footer className="border-t border-slate-200 p-4">
          <form onSubmit={handleSubmit} className="flex items-center gap-3">
            <input
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask your admissions question..."
              aria-label="Admissions question"
              disabled={loading}
              className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus:border-indigo-500 disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="rounded-xl bg-indigo-600 px-5 py-3 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Thinking..." : "Send"}
            </button>
          </form>
          <p className="mt-3 text-center text-xs text-slate-400">
            Answers are based on retrieved documents. Verify important admission
            details with official KEA notices.
          </p>
        </footer>
      </div>
    </main>
  );
}
