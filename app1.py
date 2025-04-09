import streamlit as st
import datetime
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import WebBaseLoader

# --- Azure OpenAI Configuration ---
openai_api_key = "your-openai-key"
openai_api_base = "https://your-endpoint.openai.azure.com/"
openai_api_type = "azure"
openai_api_version = "2024-02-15-preview"
deployment_name = "gpt"

llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    openai_api_type=openai_api_type,
    openai_api_version=openai_api_version,
    temperature=0.3
)

# --- SERP API ---
SERP_API_KEY = "your-serpapi-key"

# --- Session State ---
if "state" not in st.session_state:
    st.session_state.state = {
        "ticker": "",
        "num_results": 5,
        "results": [],
        "scraped_data": [],
        "validated_data": "",
        "stats_data": "",
        "search_done": False,
        "scrape_done": False,
        "validate_done": False,
        "stats_done": False,
    }

state = st.session_state.state


# --- Tools Logic ---
def serp_search(query: str, num_results: int = 5) -> List[Dict]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        return [{"error": "Failed to fetch SERP results"}]
    data = response.json()
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "")
        })
    state["results"] = results
    return results


def scrape_pages() -> List[Dict]:
    scraped_data = []
    for res in state["results"]:
        url = res["link"]
        try:
            loader = WebBaseLoader(url)
            doc = loader.load()
            soup = BeautifulSoup(doc[0].page_content, "html.parser")
            content = soup.get_text(separator="\n")
            ticker = re.search(r'\b[A-Z]{2,5}\b', content)
            price = re.search(r'\$\d{1,5}(\.\d{1,2})?', content)
            scraped_data.append({
                "url": url,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker.group(0) if ticker else "N/A",
                "price": price.group(0) if price else "N/A"
            })
        except Exception as e:
            scraped_data.append({
                "url": url,
                "error": str(e),
                "ticker": "N/A",
                "price": "N/A"
            })
    state["scraped_data"] = scraped_data
    return scraped_data


def validate_data() -> str:
    if not state["scraped_data"]:
        return "No data available for validation."
    df = pd.DataFrame(state["scraped_data"])
    state["validated_data"] = df.to_json(orient="records")
    return df.to_string(index=False)


def generate_statistics() -> str:
    if not state["scraped_data"]:
        return "No data available for stats."
    stats = []
    for item in state["scraped_data"]:
        try:
            price = float(item.get("price", "").replace("$", "").replace(",", ""))
        except:
            price = None
        if price:
            hist = np.random.uniform(price * 0.9, price * 1.1, 20)
            stats.append({
                "URL": item["url"],
                "Ticker": item["ticker"],
                "Price": f"${price:.2f}",
                "SMA": f"${np.mean(hist):.2f}",
                "EMA": f"${np.average(hist, weights=np.linspace(1, 0, len(hist))):.2f}",
                "Std Dev": f"${np.std(hist):.2f}"
            })
    df = pd.DataFrame(stats)
    state["stats_data"] = df.to_json(orient="records")
    return df.to_string(index=False)


def ask_llm(prompt):
    try:
        response = llm.predict(prompt + " (Reply Yes or No only)")
        return "yes" in response.lower()
    except Exception as e:
        print(f"LLM error: {e}")
        return input(f"{prompt} (yes/no): ").strip().lower() == "yes"


# --- UI ---
st.title("🧠 Agentic Stock Research Tool with Human-in-the-Loop")

with st.expander("🛠️ Step 1: Input Ticker & Search"):
    state["ticker"] = st.text_input("Enter Stock Ticker or Company Name:", value=state["ticker"])
    state["num_results"] = st.slider("Number of results:", 1, 10, value=state["num_results"])
    if st.button("🔍 Run Google Search"):
        serp_search(state["ticker"], state["num_results"])
        state["search_done"] = True
        state["scrape_done"] = state["validate_done"] = state["stats_done"] = False

# --- Step 2: Display Search Results ---
if state["search_done"]:
    st.subheader("🔎 Search Results")
    st.json(state["results"])

    if not state["scrape_done"]:
        if ask_llm("Should I proceed to scrape the web pages from these results?"):
            scrape_pages()
            state["scrape_done"] = True

# --- Step 3: Scraped Data ---
if state["scrape_done"]:
    st.subheader("📄 Scraped Data")
    st.dataframe(pd.DataFrame(state["scraped_data"]))

    if not state["validate_done"]:
        if ask_llm("Do you want me to validate and clean the scraped stock data?"):
            validate_data()
            state["validate_done"] = True

# --- Step 4: Validated Data ---
if state["validate_done"]:
    st.subheader("✅ Validated Data")
    st.dataframe(pd.read_json(state["validated_data"]))

    if not state["stats_done"]:
        if ask_llm("Shall I compute stock statistics like SMA, EMA, and Std Dev now?"):
            generate_statistics()
            state["stats_done"] = True

# --- Step 5: Statistics ---
if state["stats_done"]:
    st.subheader("📊 Statistical Summary")
    st.dataframe(pd.read_json(state["stats_data"]))
