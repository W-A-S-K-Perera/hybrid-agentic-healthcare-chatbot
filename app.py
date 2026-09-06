"""
app.py
"""
#import libraries
import streamlit as st
from pathlib import Path

from agent.router import handle_user_message
from agent.chat_store import (
    create_thread,
    list_threads,
    load_thread,
    save_thread,
    delete_thread,
    auto_title,
    NEW_CHAT_TITLE,
)
from agent.telemetry import summary_stats


LOGO_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "nawaloka_logo.jpg"
)


st.set_page_config(
    page_title="Nawaloka Hospital Assistant",
    page_icon=str(LOGO_PATH),
    layout="centered",
)


col1, col2 = st.columns([1, 5])

with col1:
    st.image(
        str(LOGO_PATH),
        width=70
    )

with col2:
    st.title("Nawaloka Hospital Assistant")


st.caption(
    "Ask about doctor channeling schedules, consultation fees, "
    "lab test prices, health packages, or general hospital information."
)


def _switch_thread(thread_id: str):
    loaded = load_thread(thread_id)

    st.session_state.thread_id = thread_id
    st.session_state.chat_history = loaded["messages"]
    st.session_state.memory = loaded["memory"]


if "thread_id" not in st.session_state:
    existing_threads = list_threads()

    if existing_threads:
        _switch_thread(existing_threads[0]["thread_id"])
    else:
        _switch_thread(create_thread())


with st.sidebar:

    logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])

    with logo_col2:
        st.image(
            str(LOGO_PATH),
            width=150
        )

    if st.button(
        "＋ New chat",
        use_container_width=True,
        type="primary",
    ):
        _switch_thread(create_thread())
        st.rerun()

    st.divider()

    st.caption("CHAT HISTORY")

    for thread in list_threads():

        is_active = (
            thread["thread_id"]
            == st.session_state.thread_id
        )

        row = st.columns([6, 1])

        with row[0]:

            if is_active:
                st.markdown(
                    f"""
                    <div style="
                        background-color: var(--secondary-background-color);
                        border-radius: 8px;
                        padding: 8px 10px;
                        margin-bottom: 4px;
                        font-size: 14px;
                        color: var(--text-color);
                        font-weight: 500;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    ">
                        {thread["title"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                if st.button(
                    thread["title"],
                    key=f"open_{thread['thread_id']}",
                    use_container_width=True,
                ):
                    _switch_thread(thread["thread_id"])
                    st.rerun()

        with row[1]:

            if st.button(
                "×",
                key=f"del_{thread['thread_id']}",
                help="Delete this chat",
            ):
                delete_thread(thread["thread_id"])

                remaining = list_threads()

                _switch_thread(
                    remaining[0]["thread_id"]
                    if remaining
                    else create_thread()
                )

                st.rerun()

    st.divider()

    st.subheader("About this demo")

    st.write(
        "Routes each question between:\n\n"
        "- **SQL Database** — doctor fees, schedules, lab prices, packages\n"
        "- **Vector DB (RAG)** — hospital website content, services, policies\n\n"
        "with an FAQ fast-path and persistent, multi-chat memory."
    )

    show_trace = st.checkbox(
        "Show agent reasoning trace (debug)",
        value=False
    )

    st.divider()

    st.subheader("Try asking:")

    st.markdown(
        "- *Which cardiologists are available and what's their fee?*\n"
        "- *When can I see Dr. Gunasekara?*\n"
        "- *How much is a Vitamin D test and do I need to fast?*\n"
        "- *What's included in the Well Woman package?*\n"
        "- *What services does the hospital offer?*"
    )

    st.divider()

    with st.expander("Agent performance metrics"):

        stats = summary_stats()

        if stats.get("total_interactions", 0) == 0:

            st.caption(
                "No interactions logged yet — "
                "ask a question to populate this."
            )

        else:

            st.metric(
                "Total interactions",
                stats["total_interactions"]
            )

            st.metric(
                "Success rate",
                f"{stats['success_rate']:.0%}"
            )

            st.metric(
                "Avg. latency",
                f"{stats['avg_latency_sec']}s"
            )

            st.caption("Route breakdown:")

            st.json(
                stats["route_breakdown"]
            )


def _render_sources(trace: dict):

    sources = []

    for call in trace.get("tool_calls", []):

        if (
            call["tool"] == "search_hospital_website"
            and call["result"].get("success")
        ):

            for result in call["result"].get("results", []):

                if result.get("source"):

                    sources.append(
                        (
                            result.get("title")
                            or result["source"],
                            result["source"],
                        )
                    )

    if sources:

        unique_sources = list(
            dict.fromkeys(sources)
        )

        with st.expander("📎 Sources"):

            for title, url in unique_sources:
                st.markdown(
                    f"- [{title}]({url})"
                )


for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):

        st.markdown(
            msg["content"]
        )

        if msg.get("trace"):
            _render_sources(
                msg["trace"]
            )

        if show_trace and msg.get("trace"):

            with st.expander("🔍 Agent trace"):
                st.json(
                    msg["trace"]
                )


user_input = st.chat_input(
    "Type your question..."
)


if user_input:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = handle_user_message(
                    user_input,
                    st.session_state.memory
                )

                answer = result["answer"]

            except Exception as exc:

                answer = (
                    "Sorry, I ran into an error answering that. "
                    f"Details: `{exc}`"
                )

                result = {
                    "answer": answer,
                    "error": str(exc),
                    "tool_calls": [],
                }

        st.markdown(answer)

        _render_sources(result)

        if show_trace:

            with st.expander("🔍 Agent trace"):
                st.json(result)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "trace": result,
        }
    )

    existing_title = next(
        (
            thread["title"]
            for thread in list_threads()
            if thread["thread_id"]
            == st.session_state.thread_id
        ),
        NEW_CHAT_TITLE,
    )

    if existing_title == NEW_CHAT_TITLE:
        title = auto_title(user_input)
    else:
        title = existing_title

    save_thread(
        st.session_state.thread_id,
        title,
        st.session_state.chat_history,
        st.session_state.memory,
    )
