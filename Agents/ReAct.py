from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from groq import Groq
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ------------------------
# Tool Definitions
# ------------------------
@tool
def add(a: int, b: int):
    """Adds two numbers"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtracts one number from another"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers"""
    return a * b


tools = [add, subtract, multiply]


# ------------------------
# Model Call Function using GROQ
# ------------------------
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query and use tools when needed.")

    # Convert LangChain messages to Groq-compatible format
    groq_messages = [{"role": "system", "content": system_prompt.content}]
    for msg in state["messages"]:
        # Map LangChain types to Groq roles
        role = getattr(msg, "type", "user")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role not in ["system", "user", "assistant"]:
            role = "user"

        # Handle tuple messages
        if isinstance(msg, tuple):
            groq_messages.append({"role": msg[0], "content": msg[1]})
        else:
            groq_messages.append({"role": role, "content": msg.content})

    # Call Groq API
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=groq_messages,
        temperature=1,
        max_completion_tokens=2048,
        stream=False,
    )

    # ✅ FIXED: Access via dot notation
    response = completion.choices[0].message
    ai_message = BaseMessage(content=response.content, type="ai")

    return {"messages": [ai_message]}



# ------------------------
# Conditional Logic
# ------------------------
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # Groq doesn’t have built-in tool_calls, so we simulate tool call intent
    # Detect if model asks to use tool by keyword
    if "add(" in last_message.content or "subtract(" in last_message.content or "multiply(" in last_message.content:
        return "continue"
    else:
        return "end"


# ------------------------
# Build LangGraph Flow
# ------------------------
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


# ------------------------
# Stream Output
# ------------------------
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            print("\nAssistant:", message.content)


# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    user_input = "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please."
    inputs = {"messages": [("user", user_input)]}
    print_stream(app.stream(inputs, stream_mode="values"))