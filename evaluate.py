"""
evaluate.py

"""

import argparse
import json
import time

from agent.router import handle_user_message
from agent.memory import ConversationMemory

TEST_SET = [
    {"question": "Which cardiologists are available and what's their consultation fee?", "expected": "sql"},
    {"question": "When can I see Dr. Gunasekara and in which room?", "expected": "sql"},
    {"question": "How much is a Vitamin D test and do I need to fast?", "expected": "sql"},
    {"question": "What's included in the Well Woman package and how much does it cost?", "expected": "sql"},
    {"question": "List all lab tests under the Tumor Markers category.", "expected": "sql"},
    {"question": "Which doctor has the cheapest consultation fee?", "expected": "sql"},
    {"question": "What services or medical departments does Nawaloka Hospital offer?", "expected": "vector"},
    {"question": "Tell me about the hospital's facilities and clinical centers.", "expected": "vector"},
    {"question": "What is Nawaloka Hospital's general policy on patient care?", "expected": "vector"},
    {"question": "Tell me about the cardiology department on your website, and which cardiologist can I book with?", "expected": "both"},
    {"question": "What does the website say about your ENT services, and who is the ENT surgeon and their fee?", "expected": "both"},
    {"question": "What are your visiting hours?", "expected": "faq"},
    {"question": "How do I book a channeling appointment?", "expected": "faq"},
    {"question": "Do you accept health insurance?", "expected": "faq"},
    {"question": "What's the capital of France?", "expected": "none"},
    {"question": "Can you write me a poem about the ocean?", "expected": "none"},
]


def classify_trace(trace: dict) -> str:
    """Map a router trace back to one of: sql, vector, both, faq, none."""
    if trace.get("used_faq_cache"):
        return "faq"

    tools_called = {call["tool"] for call in trace.get("tool_calls", [])}
    has_sql = "query_hospital_database" in tools_called
    has_vector = "search_hospital_website" in tools_called

    if has_sql and has_vector:
        return "both"
    if has_sql:
        return "sql"
    if has_vector:
        return "vector"
    return "none"


def run_evaluation(verbose: bool = False) -> dict:
    results = []
    correct = 0

    for i, case in enumerate(TEST_SET, 1):
        memory = ConversationMemory()  # fresh memory per test case for isolation
        start = time.time()
        trace = handle_user_message(case["question"], memory)
        elapsed = time.time() - start

        actual = classify_trace(trace)
        is_correct = actual == case["expected"]
        correct += int(is_correct)

        result = {
            "id": i,
            "question": case["question"],
            "expected": case["expected"],
            "actual": actual,
            "correct": is_correct,
            "latency_sec": round(elapsed, 2),
            "answer": trace["answer"],
        }
        results.append(result)

        status = "✅" if is_correct else "❌"
        print(f"{status} [{i}/{len(TEST_SET)}] expected={case['expected']:<8} actual={actual:<8} "
              f"({elapsed:.1f}s)  {case['question'][:60]}")
        if verbose:
            print(f"      answer: {result['answer'][:200]}")

    accuracy = correct / len(TEST_SET)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"ROUTING ACCURACY: {correct}/{len(TEST_SET)} ({accuracy:.0%})")
    print(f"AVERAGE LATENCY:  {avg_latency:.2f}s per question")
    print("=" * 60)

    return {"accuracy": accuracy, "avg_latency_sec": avg_latency, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent routing accuracy.")
    parser.add_argument("--verbose", action="store_true", help="Print each answer, not just routing.")
    parser.add_argument("--save", type=str, default=None, help="Path to save raw JSON results.")
    args = parser.parse_args()

    summary = run_evaluation(verbose=args.verbose)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Results saved to {args.save}")
