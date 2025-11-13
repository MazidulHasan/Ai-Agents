from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from groq import Groq
import os

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    messages: List[dict]

# Initialize Groq client (reads GROQ_API_KEY from environment)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process(state: AgentState) -> AgentState:
    # Get the latest user message (topic)
    user_message = state["messages"][-1]["content"]

    # Create system + user messages for Groq
    groq_messages = [
        {"role": "system", "content": "You are a professional LinkedIn post generator. Create an engaging post in 2 lines for the given topic."},
        {"role": "user", "content": user_message}
    ]

    # Request completion from Groq
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=groq_messages,
        temperature=0.9,
        max_tokens=512,
        top_p=1,
        stream=True
    )

    print("\n✨ LinkedIn Post ✨\n")
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)
    print("\n")

    return state

# Build LangGraph pipeline
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Take single input and exit
if __name__ == "__main__":
    user_input = input("Enter Topic: ").strip()
    agent.invoke({"messages": [{"role": "user", "content": user_input}]})
