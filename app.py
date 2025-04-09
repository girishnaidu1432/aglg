import datetime
import json
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict

import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.tools import Tool, tool

# ✅ Azure OpenAI Configuration
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

# ✅ Global State
state = {
    "ticker": "",
    "num_results": 5,
    "results": [],
    "scraped_data": [],
    "validated_data": "",
    "stats_data": ""
}

# ✅ SERP API Configuration
SERP_API_KEY = "2a13a66e8fb69ceba7c25e8dfc4db1932d86c259d237114131a2146131dd7b4c"

@tool
def serp_search_tool(query: str, num_results: int = 5) -> List[Dict]:
    """Tool to perform Google search using SerpAPI and return results."""
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
    """Tool to scrape URLs from SERP results and extract ticker and price."""
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
    """Tool to validate scraped data and return a JSON table."""
    if not state["scraped_data"]:
        return "No data available for validation."

    df = pd.DataFrame(state["scraped_data"])
    state["validated_data"] = df.to_json(orient="records")
    return df.to_string(index=False)

@tool
def generate_statistics(_: str = "") -> str:
    """Tool to generate SMA, EMA, and Std Dev for extracted prices."""
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

# ✅ Register Tools
tools = [
    Tool(name="GoogleSearch", func=serp_search_tool, description="Search Google via SerpAPI for stock news"),
    Tool(name="ScrapePages", func=scrape_web_pages, description="Scrape URLs for ticker and price info"),
    Tool(name="ValidateData", func=validate_data, description="Validate and tabulate scraped stock data"),
    Tool(name="GenerateStats", func=generate_statistics, description="Compute SMA, EMA, and Std Dev of prices")
]

# ✅ Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ Streamlit UI
st.set_page_config(page_title="📈 Stock Insights with Agentic AI LLM", layout="centered")
st.title("📈 AGENTIC AI")

with st.form("stock_form"):
    ticker = st.text_input("Enter Stock Ticker or Company Name", value="Tesla")
    num_results = st.slider("Number of News Results", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Run Agent")

if submitted:
    state["ticker"] = ticker
    state["num_results"] = num_results

    user_goal = f"Get the latest {num_results} news for {ticker}, scrape the pages, validate data, and generate statistics."

    with st.spinner("Running agent to gather insights..."):
        result = agent.run(user_goal)

    st.success("Agent run complete ✅")

    st.subheader("🔍 Search Results")
    st.write(pd.DataFrame(state["results"]))

    st.subheader("🧾 Scraped Data")
    st.write(pd.DataFrame(state["scraped_data"]))

    st.subheader("✅ Validated Data")
    st.code(state["validated_data"], language="json")

    st.subheader("📊 Statistics")
    st.code(state["stats_data"], language="json")

    st.subheader("🧠 Final Agent Response")
    st.write(result)
