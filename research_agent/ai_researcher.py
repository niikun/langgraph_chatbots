import os
import re
from dotenv import load_dotenv

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict


load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

summary_template = """
以下の内容を、問いへの答えを直接的に説明する簡潔な段落にまとめてください。
要点を明確かつ網羅的に盛り込みつつ、分かりやすさを維持してください。
問い:{query}
Content:{content}
"""

generate_response_template = """
以下のユーザーの質問と提示された内容をもとに、内容から関連情報を用いて直接的に回答を生成してください。
回答は明確かつ簡潔に構成し、さらに回答の要点を簡単にまとめた概要を付け加えてください。
Answerは日本語でお願いします。
質問:{question}
内容:{context}
Answer:
"""

class ResearchState(TypedDict):
    query:str
    sources:list[str]
    web_results:list[str]
    summarized_results:list[str]
    response:str

class ReseachStateInput(TypedDict):
    query:str

class ResearchStateOutput(TypedDict):
    sources:list[str]
    response:str

def search_web(state:ResearchState):
    """
    Perform a web search using the Tavily API and update the state with the results.
    """
    search = TavilySearchResults(max_results=3)
    search_resutls = search.invoke(state["query"])
    print(search_resutls)
    return {
        "sources":[result["url"] for result in search_resutls],
        "web_results":[result["content"] for result in search_resutls]
    }


def summarize_results(state:ResearchState):
    model = ChatOllama(model="deepseek-r1:1.5b")
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model
    summarized_results = []
    for content in state["web_results"]:
        summary = chain.invoke({"query":state["query"],"content":content})
        text = summary.content if hasattr(summary, "content") else str(summary)
        clean_content = clean_text(text)
        summarized_results.append(clean_content)
    print(summarized_results)
    return {
        "summarized_results":summarized_results
    }

def generate_response(state:ResearchState):
    model = ChatOllama(model="deepseek-r1:1.5b")
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model
    content = "\n\n".join([summary for summary in state["summarized_results"]])

    response = chain.invoke({"question":state["query"],"context":content})
    print(response)
    # response.contentが存在する場合はそれを使う
    text = response.content if hasattr(response, "content") else str(response)
    return {
        "response": text
    }

def clean_text(text:str):
    cleaned_text = re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
    return cleaned_text

builder = StateGraph(
    ResearchState,
    input=ReseachStateInput, 
    output=ResearchStateOutput
    ) 

builder.add_node("serach_web",search_web)
builder.add_node("summarize_results",summarize_results)
builder.add_node("generate_response",generate_response)
builder.add_edge(START,"serach_web")
builder.add_edge("serach_web","summarize_results")
builder.add_edge("summarize_results","generate_response")
builder.add_edge("generate_response",END)

graph = builder.compile()


st.title("AI Researcher")
query = st.text_input("質問してください")

if query:
    response_state = graph.invoke({"query":query})
    st.write(clean_text(response_state["response"]))

    st.subheader("Sources")
    for source in response_state["sources"]:
        st.write(source)