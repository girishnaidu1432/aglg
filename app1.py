import streamlit as st
import datetime
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import WebBaseLoader

# --- Azure OpenAI Configuration ---
openai_api_key = "your-openai-key"
openai_api_base = "https://your-azure-openai-endpoint/"
openai_api_type = "azure"
openai_api_version = "2024-02-15-preview"
deployment_name = "gpt"

llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    openai_api_type=openai_api_type,
    openai_api_version=openai_api_version,
    temperature=0.5
)

# --- SERP API Key ---
SERP_API_KEY = "your-serp-api-key"

# --- Initialize Session State ---
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
        "stats_done": False
    }

state = st.session_state.state

# --- Tools Definition ---
@tool
def serp_search_tool(query: str, num_results: int = 5) -> List[Dict]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        return [{"error": "Failed to fetch SERP API results"}]
    data = response.json()
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        title = item.get("title", "No Title")
        link = item.get("link", "No Link")
        snippet = item.get("snippet", "No Snippet")
        results.append({"title": title, "link": link, "snippet": snippet})
    state["results"] = results
    return results

@tool
def scrape_web_pages(_: str = "") -> List[Dict]:
    scraped_data = []
    for res in state["results"]:
        url = res["link"]
        try:
            loader = WebBaseLoader(url)
            doc = loader.load()
            soup = BeautifulSoup(doc[0].page_content, "html.parser")
            full_content = soup.get_text(separator="\n")
            ticker_match = re.search(r'\b[A-Z]{2,5}\b', full_content)
            ticker = ticker_match.group(0) if ticker_match else "N/A"
            price_match = re.search(r'\$\d{1,5}(\.\d{1,2})?', full_content)
            price = price_match.group(0) if price_match else "N/A"
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scraped_data.append({"url": url, "date": date, "ticker": ticker, "price": price})
        except Exception as e:
            scraped_data.append({"url": url, "error": str(e), "ticker": "N/A", "price": "N/A"})
    state["scraped_data"] = scraped_data
    return scraped_data

@tool
def validate_data(_: str = "") -> str:
    if not state["scraped_data"]:
        return "No data available for validation."
    df = pd.DataFrame(state["scraped_data"])
    state["validated_data"] = df.to_json(orient="records")
    return df.to_string(index=False)

@tool
def generate_statistics(_: str = "") -> str:
    if not state["scraped_data"]:
        return "No data available for stats."
    stats = []
    for item in state["scraped_data"]:
        try:
            price = float(item.get("price", "N/A").replace("$", "").replace(",", ""))
        except:
            price = None
        if price:
            hist_prices = np.random.uniform(low=price * 0.9, high=price * 1.1, size=20)
            sma = np.mean(hist_prices)
            ema = np.average(hist_prices, weights=np.linspace(1, 0, len(hist_prices)))
            std_dev = np.std(hist_prices)
            stats.append({
                "URL": item["url"],
                "Ticker": item["ticker"],
                "Price": f"${price:.2f}",
                "SMA": f"${sma:.2f}",
                "EMA": f"${ema:.2f}",
                "Std Dev": f"${std_dev:.2f}"
            })
    df_stats = pd.DataFrame(stats)
    state["stats_data"] = df_stats.to_json(orient="records")
    return df_stats.to_string(index=False)

# --- UI Layout ---
st.title("📊 Agentic Stock Insight Tool with Human-in-the-Loop")

# --- Step 1: User Input ---
with st.expander("🔧 Input Configuration", expanded=True):
    state["ticker"] = st.text_input("Enter stock ticker or company name:", value=state.get("ticker", ""))
    state["num_results"] = st.number_input("Number of search results:", min_value=1, max_value=10, value=state.get("num_results", 5))
    if st.button("🔍 Run Google Search"):
        serp_search_tool(state["ticker"], state["num_results"])
        state["search_done"] = True

# --- Step 2: Show Search Results ---
if state["search_done"]:
    st.subheader("🔍 Google Search Results")
    st.json(state["results"])

# --- Step 3: Human-in-the-loop for Scraping ---
if state["search_done"]:
    scrape_decision = st.radio("📥 Proceed to scrape these web pages?", ("Yes", "No"), key="scrape")
    if scrape_decision == "Yes" and not state["scrape_done"]:
        scrape_web_pages()
        state["scrape_done"] = True

if state["scrape_done"]:
    st.subheader("🧾 Scraped Data")
    st.dataframe(pd.DataFrame(state["scraped_data"]))

# --- Step 4: Human-in-the-loop for Validation ---
if state["scrape_done"]:
    validate_decision = st.radio("✅ Proceed to validate scraped data?", ("Yes", "No"), key="validate")
    if validate_decision == "Yes" and not state["validate_done"]:
        validate_data()
        state["validate_done"] = True

if state["validate_done"]:
    st.subheader("📋 Validated Table")
    df = pd.read_json(state["validated_data"])
    st.dataframe(df)

# --- Step 5: Human-in-the-loop for Stats ---
if state["validate_done"]:
    stats_decision = st.radio("📈 Generate statistical insights (SMA, EMA, Std Dev)?", ("Yes", "No"), key="stats")
    if stats_decision == "Yes" and not state["stats_done"]:
        generate_statistics()
        state["stats_done"] = True

if state["stats_done"]:
    st.subheader("📊 Statistical Summary")
    df_stats = pd.read_json(state["stats_data"])
    st.dataframe(df_stats)
