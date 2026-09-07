# Nawaloka Hybrid Agentic Healthcare Chatbot

This project is a Hybrid Agentic Healthcare Chatbot developed for the Associate AI Engineer technical assignment.

The chatbot can understand a user's question and decide where the required information should come from. It can use the hospital's structured SQL database, search information collected from the Nawaloka Hospital website, or use both sources when necessary.

The application also includes a simple Streamlit chat interface, FAQ handling, conversation memory, persistent chat history, source display, and basic agent performance monitoring.

## What the chatbot can do

The chatbot is designed to answer questions about:

- Doctor availability and channeling schedules
- Doctor consultation fees
- Laboratory test prices
- Health packages
- Hospital departments and services
- General hospital information
- Frequently asked questions

For example:

```text
Which cardiologists are available and what's their fee?
```

This question requires structured information from the SQL database.

```text
What services does the hospital offer?
```

This question is handled using information retrieved from the hospital website.

The chatbot can also handle questions that require both sources:

```text
Tell me about the cardiology department and which cardiologists I can book.
```

## Architecture

```text
                         User
                           |
                           v
                  +-----------------+
                  |  Streamlit UI   |
                  |     app.py      |
                  +--------+--------+
                           |
                           v
                  +-----------------+
                  |  FAQ Fast Path  |
                  +--------+--------+
                           |
                    FAQ match?
                     /       \
                   Yes        No
                   |           |
                   v           v
              FAQ Answer   Agent Router
                               |
                     +---------+---------+
                     |                   |
                     v                   v
                SQL Tool          Vector Search
                     |                   |
                     v                   v
              hospital.db       Chroma Vector DB
                     |                   |
                     +---------+---------+
                               |
                               v
                         Final Answer
```

The agent router uses Gemini function/tool calling to decide which tool should be used. This allows the chatbot to handle questions that require one source as well as questions that require information from both sources.

## Project Structure

```text
nawaloka-hybrid-agentic-healthcare-chatbot/
|
├── app.py
├── evaluate.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
|
├── agent/
│   ├── router.py
│   ├── sql_tool.py
│   ├── vector_tool.py
│   ├── faq_cache.py
│   ├── memory.py
│   ├── chat_store.py
│   └── telemetry.py
|
├── scripts/
│   ├── setup_database.py
│   └── scrape_website.py
|
├── tests/
│   ├── test_sql_tool.py
│   ├── test_memory.py
│   └── test_chat_store.py
|
├── data/
│   └── data.sql
|
└── assets/
    └── nawaloka_logo.jpg
```

## Main Components

### Agent Router

`agent/router.py`

This is the main orchestration layer of the chatbot.

It uses Gemini to understand the user's question and decide which tool or tools should be called.

The router can:

- Call the SQL tool
- Call the website/vector search tool
- Call multiple tools for combined questions
- Handle multiple tool-calling rounds
- Maintain conversation context
- Retry requests when rate limits occur
- Return the final response to the user

### SQL Tool

`agent/sql_tool.py`

The SQL tool handles structured hospital information.

It uses SQLite and provides information such as:

- Doctors
- Specialties
- Channeling sessions
- Laboratory tests
- Health packages

The SQL tool is read-only. Only `SELECT` and `WITH` queries are allowed, while destructive SQL operations and statement stacking are blocked.

This provides an additional safety layer around LLM-generated SQL.

### Vector Search Tool

`agent/vector_tool.py`

The vector search tool is used for unstructured information collected from the Nawaloka Hospital website.

The website content is:

1. Scraped
2. Cleaned
3. Split into smaller chunks
4. Converted into embeddings
5. Stored in ChromaDB

The retrieval process uses embedding similarity together with keyword matching to improve the relevance of retrieved content.

### FAQ Fast Path

`agent/faq_cache.py`

Frequently asked questions are handled through a small FAQ cache.

Before sending every question to the LLM, the system checks whether the question is similar to one of the predefined FAQs.

This helps reduce unnecessary LLM calls and provides faster responses for common questions.

### Conversation Memory

`agent/memory.py`

The chatbot maintains conversation context so that follow-up questions can be understood correctly.

The memory uses:

- A short-term conversation buffer
- A rolling summary for older conversation history

When the conversation becomes longer, older information can be summarized instead of continuously adding the complete conversation to the prompt.

### Chat History

`agent/chat_store.py`

Chat conversations are stored locally using SQLite.

The Streamlit sidebar provides:

- New chat
- Previous conversations
- Automatically generated chat titles
- Chat switching
- Chat deletion
- Persistent conversations after restarting the application

This is intended for the local demo environment. There is no authentication system in this assignment.

### Telemetry

`agent/telemetry.py`

Basic interaction information is recorded for monitoring the chatbot.

The application can display:

- Total interactions
- Success rate
- Average response latency
- Route breakdown

## Requirements

The project requires:

- Python 3.10 or later
- Gemini API key
- Internet connection for website scraping
- Python packages listed in `requirements.txt`

Main technologies used:

- Python
- Streamlit
- Gemini
- SQLite
- ChromaDB
- Sentence Transformers
- Playwright
- Pytest

## Environment Setup

Follow the steps below to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/W-A-S-K-Perera/hybrid-agentic-healthcare-chatbot.git
cd hybrid-agentic-healthcare-chatbot
```

### 2. Create a virtual environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the Playwright browser required by the website scraper:

```bash
playwright install
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root.

You can copy `.env.example` and rename it to `.env`.

Then add your API key:

```env
GEMINI_API_KEY=your_api_key_here
```

A Gemini API key can be created using Google AI Studio:

https://aistudio.google.com/app/apikey



## Database Setup

The provided `data.sql` file contains the structured hospital information used by the chatbot.

Run:

```bash
python scripts/setup_database.py
```

This creates the local SQLite database:

```text
data/hospital.db
```

The database contains information such as:

- Doctors
- Specialties
- Channeling sessions
- Laboratory tests
- Health packages

The setup script also provides information about the imported tables and rows so the database setup can be verified.

## Website and Vector Database Setup

The chatbot uses information from the Nawaloka Hospital website as its unstructured knowledge source.

Run:

```bash
python scripts/scrape_website.py
```

The scraper collects pages from:

```text
https://www.nawaloka.com
```

The collected content is cleaned and divided into smaller chunks before being converted into embeddings.

The embeddings are stored in:

```text
vectorstore/chroma_db
```

The first run may take some time because the Sentence Transformers embedding model needs to be downloaded.

An internet connection is required for this step.

If the website content changes, the scraper can be run again to refresh the vector database.

## Running the Application

After completing the database and vector database setup, start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open the URL in a browser to use the chatbot.

## Example Questions

Here are some questions that can be used to test the chatbot.

### SQL questions

```text
Which cardiologists are available and what's their fee?
```

```text
When can I see Dr. Gunasekara?
```

```text
How much is a Vitamin D test?
```

```text
What's included in the Well Woman package?
```

### Website questions

```text
What services and departments does the hospital have?
```

```text
Tell me about the hospital facilities.
```

### Combined questions

```text
Tell me about the cardiology department and which cardiologists I can book.
```

The agent should use the appropriate source based on the question.

## Chat History and Memory

The application supports multiple conversations through the sidebar.

A new conversation can be created using the `+ New chat` button.

Previous conversations are stored locally and can be reopened after restarting the application.

The first user message is used to generate a simple title for the conversation.

The conversation memory keeps recent messages available while older information can be summarized when the conversation becomes long.

## Agent Trace

For demonstration and debugging, the sidebar includes an option to show the agent trace.

When enabled, the application can show information such as:

- Which tool was selected
- SQL queries generated by the agent
- Vector search queries
- Tool results
- Routing information

This is useful when demonstrating the agent's routing behaviour.

## Evaluation

The project includes `evaluate.py` to evaluate the routing behaviour of the agent.

Run:

```bash
python evaluate.py
```

The evaluation includes different types of questions:

- SQL-only questions
- Vector search questions
- Questions requiring both sources
- FAQ questions
- Off-topic questions

The script reports routing accuracy and response latency.

Example output:

```text
ROUTING ACCURACY: 15/16 (94%)
AVERAGE LATENCY: 1.8s per question
```

The exact result may vary depending on the LLM response and runtime environment.

## Running Tests

Unit tests are included for the main supporting components.

Run:

```bash
pytest tests/ -v
```

The tests cover areas such as:

- SQL safety guardrails
- Invalid SQL handling
- Conversation memory
- Memory compaction
- Chat creation
- Chat persistence
- Chat deletion

## Design Decisions

### Why SQLite?

SQLite was selected because it is simple to set up and does not require a separate database server.

This makes the project easier to run and evaluate locally while still providing the SQL functionality required by the assignment.

The database setup is kept inside `scripts/setup_database.py`, which also makes it easier to replace SQLite with another SQL database in the future.

### Why local embeddings?

The vector database uses a local Sentence Transformers model for embeddings.

This avoids requiring a separate embedding API key and keeps the retrieval pipeline simple to run.

### Why use an agent router?

A fixed keyword-based router would only work for a limited set of questions.

Instead, the LLM is responsible for understanding the user's intent and selecting the appropriate tool.

This is particularly useful for questions that require information from both the SQL database and the hospital website.

## Additional Features

The project includes several features beyond the basic requirements:

- FAQ fast path for common questions
- Hybrid vector retrieval
- Conversation memory
- Persistent multi-chat history
- SQL safety guardrails
- Agent routing evaluation
- Performance telemetry
- Source display for website-based answers
- Rate-limit retry and backoff
- Unit tests
- Agent debugging trace


## Future Improvements

If the project were developed further, possible improvements would include:

- User authentication and role-based access
- PostgreSQL or another production database
- A more advanced website crawler
- Document upload and retrieval
- Dedicated retrieval reranking
- Larger evaluation datasets
- Automated monitoring and tracing
- User-specific conversation storage
- Additional healthcare safety checks
- Cloud deployment


## Author

Sanduni Perera

