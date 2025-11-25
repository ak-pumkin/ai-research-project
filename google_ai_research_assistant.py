# google_ai_research_assistant.py
import os
import logging
from googleapiclient.discovery import build
from google.generativeai import TextGenerationClient
import streamlit as st
from dotenv import load_dotenv

# ------------------- Load Environment Variables ------------------- #
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
PALM_API_KEY = os.getenv("PALM_API_KEY")

# ------------------- Logging / Observability ------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------- Google Services ------------------- #
search_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
palm_client = TextGenerationClient(api_key=PALM_API_KEY)

# ------------------- Session & Long-Term Memory ------------------- #
session_memory = {}
long_term_memory = {}

def save_session(user, query, output):
    if user not in session_memory:
        session_memory[user] = []
    session_memory[user].append({"query": query, "output": output})
    logger.info(f"Session memory updated for user: {user}")

def save_long_term(user, query, output):
    if user not in long_term_memory:
        long_term_memory[user] = []
    long_term_memory[user].append({"query": query, "output": output})
    logger.info(f"Long-term memory updated for user: {user}")

# ------------------- Tools / Agents ------------------- #

# Summarizer Agent (Sequential)
def summarize_text(text):
    prompt = f"Summarize this research text in 3-4 concise sentences:\n{text}"
    response = palm_client.generate_text(model="models/text-bison-001", prompt=prompt)
    logger.info("Summarization completed")
    return response.text

# Fact-Checker Agent (Loop)
def fact_check(statement):
    try:
        res = search_service.cse().list(q=statement, cx=GOOGLE_CSE_ID, num=3).execute()
        results = [item['snippet'] for item in res.get('items', [])]
        search_summary = "\n".join(results)
        prompt = f"Fact-check this statement: '{statement}' using these sources:\n{search_summary}\nProvide True/False and a short explanation."
        response = palm_client.generate_text(model="models/text-bison-001", prompt=prompt)
        logger.info("Fact-check completed")
        return response.text
    except Exception as e:
        logger.error(f"Error in fact-checking: {e}")
        return "Error occurred during fact-checking."

# Code Generator Agent (Parallel)
def generate_code(task):
    prompt = f"Generate Python code for the following task:\n{task}"
    response = palm_client.generate_text(model="models/code-bison-001", prompt=prompt)
    logger.info("Code generation completed")
    return response.text

# ------------------- Multi-Agent Coordinator ------------------- #
def research_assistant(user, query, mode="summarize"):
    """
    Main coordinator for multi-agent research assistant
    Modes: summarize, factcheck, code
    """
    if mode == "summarize":
        output = summarize_text(query)
    elif mode == "factcheck":
        output = fact_check(query)
    elif mode == "code":
        output = generate_code(query)
    else:
        output = "Invalid mode. Choose 'summarize', 'factcheck', or 'code'."
        logger.warning(f"Invalid mode selected: {mode}")
    
    # Save memory
    save_session(user, query, output)
    save_long_term(user, query, output)
    return output

# ------------------- Streamlit Web Interface ------------------- #
st.set_page_config(page_title="Google AI Research Assistant", layout="wide")
st.title("🧠 Google AI Research Assistant (Multi-Agent)")

# User Input
user_id = st.text_input("Enter your name:", "")
mode = st.selectbox("Choose Mode:", ["summarize", "factcheck", "code"])
user_input = st.text_area("Enter your query here:")

# Run Button
if st.button("Run"):
    if not user_id:
        st.error("Please enter your name to track sessions!")
    elif not user_input:
        st.error("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            result = research_assistant(user_id, user_input, mode)
        st.success("✅ Done!")
        st.write(result)

        # Display Memories
        st.subheader("📝 Session Memory")
        st.write(session_memory.get(user_id, []))

        st.subheader("💾 Long-Term Memory")
        st.write(long_term_memory.get(user_id, []))
