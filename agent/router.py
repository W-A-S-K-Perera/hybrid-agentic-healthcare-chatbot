"""
router.py
Routes user questions to the FAQ, SQL database, vector store, or a combination of tools, then generates the final response.
"""

import json
import os
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from dotenv import load_dotenv

from agent.sql_tool import run_sql_query, SQL_TOOL_SPEC
from agent.vector_tool import search_hospital_website, VECTOR_TOOL_SPEC
from agent.faq_cache import match_faq
from agent.memory import ConversationMemory
from agent.telemetry import log_interaction

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_TOOL_ROUNDS = 4
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 2


class RateLimitedError(RuntimeError):
    """Raised when retries are exhausted due to provider rate limiting."""


def _call_with_retry(fn, *args, **kwargs):
    """
    Call an LLM SDK function with graceful handling of rate limits and
    transient outages -- required by the assignment ("manage
    rate-limiting gracefully").

    - On a 429 (ResourceExhausted), respects the provider's suggested
      `retry_delay` if present, otherwise backs off exponentially.
    - On a transient 5xx (ServiceUnavailable), backs off and retries.
    - After MAX_RETRIES failed attempts, raises RateLimitedError with a
      clear, user-facing message rather than a raw stack trace.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except ResourceExhausted as exc:
            last_exc = exc
            retry_after = _extract_retry_delay(exc) or (BASE_BACKOFF_SEC * attempt)
            if attempt < MAX_RETRIES:
                time.sleep(retry_after)
        except ServiceUnavailable as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(BASE_BACKOFF_SEC * attempt)

    raise RateLimitedError(
        "The assistant is temporarily rate-limited by the LLM provider's free tier "
        "and couldn't get a response after several retries. Please wait a bit and "
        "try again, or switch GEMINI_MODEL in .env to a model with a higher free "
        f"quota. (Underlying error: {last_exc})"
    )


def _extract_retry_delay(exc) -> float | None:
    """Best-effort extraction of the provider's suggested retry delay (seconds)."""
    try:
        for detail in getattr(exc, "details", lambda: [])():
            if hasattr(detail, "retry_delay"):
                return detail.retry_delay.seconds
    except Exception:  # noqa: BLE001 - fall back to exponential backoff
        pass
    return None


genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant for Nawaloka Hospital's patient chatbot.

You have two tools:
1. query_hospital_database - for structured/transactional data: doctor
   fees, channeling schedules, room numbers, lab test prices/prep
   instructions, health package prices and inclusions.
2. search_hospital_website - for unstructured/informational content:
   hospital services, departments, general policies, facilities.

Rules:
- If a question needs BOTH kinds of info (e.g. "which cardiologist can I
  see and what services does the cardiology dept offer?"), call both
  tools before answering.
- Never invent prices, schedules, or doctor details -- always look them
  up with query_hospital_database. If a query returns no rows, say so
  plainly instead of guessing.
- Keep answers concise, warm, and easy for a patient to read. Use
  bullet points for lists of options (doctors, tests, packages).
- If asked something outside hospital-related topics, politely say you
  can only help with hospital-related questions.
- Prices are for informational demo purposes; remind the user to
  confirm the final price with the hospital when it's relevant (e.g.
  booking a package).
"""

_TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name=SQL_TOOL_SPEC["name"],
                description=SQL_TOOL_SPEC["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "sql": genai.protos.Schema(type=genai.protos.Type.STRING)
                    },
                    required=["sql"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name=VECTOR_TOOL_SPEC["name"],
                description=VECTOR_TOOL_SPEC["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "query": genai.protos.Schema(type=genai.protos.Type.STRING)
                    },
                    required=["query"],
                ),
            ),
        ]
    )
]

_TOOL_IMPLS = {
    "query_hospital_database": lambda args: run_sql_query(args["sql"]),
    "search_hospital_website": lambda args: search_hospital_website(args["query"]),
}


def _summarize_for_memory(existing_summary: str, turns: list[dict]) -> str:
    """LLM-backed compaction callback used by ConversationMemory.compact()."""
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    prompt = (
        "Update the running summary of this patient support conversation. "
        "Keep it short (3-5 sentences), factual, and preserve any patient "
        "preferences or facts mentioned (e.g. doctor names discussed, symptoms, "
        "which department they're interested in).\n\n"
        f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
        f"New turns to fold in:\n{transcript}\n\n"
        "Updated summary:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def handle_user_message(user_message: str, memory: ConversationMemory) -> dict:
    """
    Main entry point called by the UI.

    Returns:
        {
          "answer": str,
          "used_faq_cache": bool,
          "tool_calls": [ {"tool": str, "args": dict, "result": dict}, ... ],
        }
    """
    start_time = time.time()
    trace = {"used_faq_cache": False, "tool_calls": []}

    try:
        # 1) FAQ fast path
        faq_answer = match_faq(user_message)
        if faq_answer:
            trace["used_faq_cache"] = True
            memory.add_turn("user", user_message)
            memory.add_turn("assistant", faq_answer)
            trace["answer"] = faq_answer
            log_interaction(user_message, trace, time.time() - start_time)
            return trace

        # 2) Full agent path with tool calling
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
            tools=_TOOLS,
        )

        context = memory.as_context_string()
        initial_prompt = (
            f"{context}\n\nCurrent user question: {user_message}" if context else user_message
        )

        chat = model.start_chat()
        response = _call_with_retry(chat.send_message, initial_prompt)

        for _ in range(MAX_TOOL_ROUNDS):
            function_calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if part.function_call
            ]
            if not function_calls:
                break

            tool_responses = []
            for fc in function_calls:
                tool_name = fc.name
                args = dict(fc.args)
                impl = _TOOL_IMPLS.get(tool_name)
                result = impl(args) if impl else {"success": False, "error": f"Unknown tool {tool_name}"}
                trace["tool_calls"].append({"tool": tool_name, "args": args, "result": result})
                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": json.dumps(result)},
                        )
                    )
                )
            response = _call_with_retry(chat.send_message, genai.protos.Content(parts=tool_responses))

        final_answer = response.text.strip()

        # 3) Update + compact memory
        memory.add_turn("user", user_message)
        memory.add_turn("assistant", final_answer)
        if memory.needs_compaction():
            memory.compact(_summarize_for_memory)

        trace["answer"] = final_answer
        log_interaction(user_message, trace, time.time() - start_time)
        return trace

    except Exception as exc:  # noqa: BLE001 - log the failure, then let the UI show it
        log_interaction(user_message, trace, time.time() - start_time, error=str(exc))
        raise
