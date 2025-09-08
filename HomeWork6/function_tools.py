from typing import Annotated
from sympy import sympify
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from arxiv_search import search_arxiv as arxiv_search_func
import json


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    Use this tool when the user asks for mathematical calculations.
    
    Args:
        expression (str): Mathematical expression to evaluate (e.g., "2+2", "sqrt(16)", "sin(pi/2)")
        
    Returns:
        str: The result of the calculation or error message
    """
    try:
        result = sympify(expression)
        return f"The result of '{expression}' is: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def search_arxiv(query: str) -> str:
    """
    Search arXiv for academic papers related to the given query.
    Use this tool when the user asks about research papers, scientific topics, or academic literature.
    
    Args:
        query (str): Search query for arXiv papers (e.g., "quantum computing", "machine learning")
        
    Returns:
        str: Formatted results containing paper titles, authors, abstracts, and links
    """
    return arxiv_search_func(query)


# List of available tools
tools = [calculate, search_arxiv]


def should_continue(state: MessagesState) -> str:
    """Determine if we should continue to tools or end."""
    messages = state['messages']
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


def call_model(state: MessagesState, model):
    """Call the LLM model with the current state."""
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


class FunctionCallingAgent:
    """LangGraph-based agent that can call functions based on user queries."""
    
    def __init__(self, model):
        """
        Initialize the function calling agent.
        
        Args:
            model: The LLM model to use (should support tool calling)
        """
        self.model = model.bind_tools(tools)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Define a new graph
        workflow = StateGraph(MessagesState)
        
        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: call_model(state, self.model))
        workflow.add_node("tools", ToolNode(tools))
        
        # Set the entrypoint as `agent`
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        return workflow.compile(checkpointer=self.memory)
    
    def process_query(self, user_input: str, thread_id: str = "default") -> str:
        """
        Process a user query and return the response.
        
        Args:
            user_input (str): The user's input text
            thread_id (str): Thread ID for conversation memory
            
        Returns:
            str: The agent's response
        """
        try:
            # Create the initial message
            initial_message = HumanMessage(content=user_input)
            
            # Run the graph
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(
                {"messages": [initial_message]}, 
                config
            )
            
            # Get the last message (the response)
            last_message = result["messages"][-1]
            
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return str(last_message)
                
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_conversation_history(self, thread_id: str = "default") -> list:
        """
        Get the conversation history for a given thread.
        
        Args:
            thread_id (str): Thread ID to retrieve history for
            
        Returns:
            list: List of messages in the conversation
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.graph.get_state(config)
            return snapshot.values.get("messages", [])
        except Exception as e:
            return []


def create_agent(model):
    """
    Factory function to create a function calling agent.
    
    Args:
        model: The LLM model to use
        
    Returns:
        FunctionCallingAgent: Configured agent instance
    """
    return FunctionCallingAgent(model)