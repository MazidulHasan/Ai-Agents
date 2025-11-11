from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the agent state
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Define the process node
def process(state: AgentState) -> AgentState:
    """This node will process the user input and respond using Groq API (streaming enabled)."""

    # Convert LangChain-style messages to Groq format
    messages_for_groq = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in state["messages"]
    ]

    print(f"messages_for_groq :: {messages_for_groq}")

    # Create the completion (streaming)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages_for_groq,
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )

    ai_content = ""
    print("\nAI:", end=" ")

    for chunk in completion:
        if chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            ai_content += piece
            print(piece, end="", flush=True)

    print("\n")

    # Append AI message to state
    state["messages"].append(AIMessage(content=ai_content))
    print("CURRENT STATE:", state["messages"])

    return state


# Build LangGraph pipeline
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Conversation history
conversation_history: List[Union[HumanMessage, AIMessage]] = []

# Interactive loop
user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")



# Define target folder and ensure it exists
log_folder = os.path.join("Ai-Agents", "Agents")
os.makedirs(log_folder, exist_ok=True)

# Define full path for the log file
log_path = os.path.join(log_folder, "logging.txt")

# Save conversation to file
with open(log_path, "w", encoding="utf-8") as file:
    file.write("Your Conversation Log:\n\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print(f"\n✅ Conversation saved to {log_path}")

