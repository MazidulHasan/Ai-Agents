from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from groq import Groq
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[dict]

# Initialize Groq client (reads GROQ_API_KEY from environment)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process(state: AgentState) -> AgentState:
    # create completion using Groq
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=state["messages"],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )

    print("\nAI:", end=" ")
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

    return state

#Build LangGraph pipeline
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Interactive console loop
user_input = input("Enter: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    user_input = input("Enter: ")
