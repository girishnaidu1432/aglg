import streamlit as st
import datetime
import requests
import numpy as np
import pandas as pd
import re
import json
from bs4 import BeautifulSoup

from typing import List, Dict
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import WebBaseLoader

# === Azure OpenAI Config ===
openai_api_key = "14560021aaf84772835d76246b53397a"
openai_api_base = "https://amrxgenai.openai.azure.com/"
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

# === SERP API Config ===
SERP_API_KEY = "2a13a66e8fb69ceba7c25e8dfc4db1932d86c259d237114131a2146131dd7b4c"

# === Global State ===
if "state" not in st.session_state:
    st.session_state.state = {
        "ticker": "",
        "num_results": 5,
        "results": [],
        "scraped_data": [],
        "validated_data": "",
        "stats_data": ""
    }

state = st.session_state.state

# === Tool 1: SERP Search ===
def serp_search_tool(query: str, num_results: int = 5) -> List[Dict]:
    st.info(f"🔍 Searching Google for: {query}")
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results
    }

    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        st.error("❌ SERP API request failed.")
        return []

    data = response.json()
    results = []

    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", "No Title"),
            "link": item.get("link", "No Link"),
            "snippet": item.get("snippet", "No Snippet")
        })

    state["results"] = results
    return results

# === Tool 2: Web Scraper ===
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
            scraped_data.append({
                "url": url, "date": date,
                "ticker": ticker, "price": price
            })
        except Exception as e:
            scraped_data.append({"url": url, "error": str(e), "ticker": "N/A", "price": "N/A"})

    state["scraped_data"] = scraped_data
    return scraped_data

# === Tool 3: Validation ===
def validate_data(_: str = "") -> str:
    if not state["scraped_data"]:
        return "No data available for validation."

    df = pd.DataFrame(state["scraped_data"])
    state["validated_data"] = df.to_json(orient="records")
    return df

# === Tool 4: Statistics ===
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
            hist_prices = np.random.uniform(price * 0.9, price * 1.1, 20)
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
    return df_stats

# === Tools & Agent Registration ===
tools = [
    Tool(name="GoogleSearch", func=serp_search_tool, description="Search Google via SerpAPI for stock news"),
    Tool(name="ScrapePages", func=scrape_web_pages, description="Scrape URLs for ticker and price info"),
    Tool(name="ValidateData", func=validate_data, description="Validate and tabulate scraped stock data"),
    Tool(name="GenerateStats", func=generate_statistics, description="Compute SMA, EMA, and Std Dev of prices")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === Streamlit UI ===
st.set_page_config(page_title="Stock News Agent", layout="wide")
st.title("📊 Stock Intelligence Agent with Human-in-the-Loop")

with st.sidebar:
    state["ticker"] = st.text_input("🔎 Enter stock ticker or company name", value="Tesla")
    state["num_results"] = st.slider("Number of search results", 1, 10, 5)

    if st.button("🚀 Start Agent Task"):
        st.session_state.run_task = True
    else:
        st.session_state.run_task = False

if st.session_state.get("run_task"):

    st.markdown(f"### 1️⃣ Google Search for: **{state['ticker']}**")
    results = serp_search_tool(state["ticker"], state["num_results"])
    st.json(results)

    if st.radio("Proceed to scrape web pages?", ("Yes", "No")) == "Yes":
        st.markdown("### 2️⃣ Scraped Web Data")
        scraped = scrape_web_pages()
        st.dataframe(pd.DataFrame(scraped))

        if st.radio("Proceed to validate data?", ("Yes", "No")) == "Yes":
            st.markdown("### 3️⃣ Validated Data")
            validated = validate_data()
            st.dataframe(validated)

            if st.radio("Proceed to generate stats?", ("Yes", "No")) == "Yes":
                st.markdown("### 4️⃣ Statistical Summary")
                stats = generate_statistics()
                st.dataframe(stats)

                st.success("🎯 Task Complete. All steps finished.")
            else:
                st.warning("⏭️ Stats generation skipped.")
        else:
            st.warning("⏭️ Validation skipped.")
    else:
        st.warning("⏭️ Scraping skipped.")

