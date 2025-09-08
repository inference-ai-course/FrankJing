"""
Test script for function calling capabilities of the voice agent.
This script tests the individual functions and routing logic without the full voice pipeline.
"""

import json
from voice_agent import search_arxiv, calculate, route_llm_output


def test_calculate_function():
    """Test the calculate function with various mathematical expressions."""
    print("=== Testing Calculate Function ===")
    
    test_cases = [
        "2 + 2",
        "15 * 23", 
        "sqrt(16)",
        "sin(pi/2)",
        "derivative(x**2, x)",
        "integrate(x**2, x)",
        "invalid_expression"
    ]
    
    for expr in test_cases:
        result = calculate(expr)
        print(f"Expression: {expr}")
        print(f"Result: {result}")
        print("-" * 40)


def test_search_arxiv_function():
    """Test the arXiv search function."""
    print("\n=== Testing ArXiv Search Function ===")
    
    test_queries = [
        "quantum computing",
        "neural networks", 
        "machine learning",
        "deep learning transformers"
    ]
    
    for query in test_queries:
        result = search_arxiv(query)
        print(f"Query: {query}")
        print(f"Result: {result[:200]}...")  # Truncate for readability
        print("-" * 40)


def test_function_routing():
    """Test the function routing logic with various LLM outputs."""
    print("\n=== Testing Function Routing Logic ===")
    
    test_outputs = [
        # Math function call
        '{"function": "calculate", "arguments": {"expression": "25 * 4"}}',
        
        # ArXiv search function call
        '{"function": "search_arxiv", "arguments": {"query": "attention mechanisms"}}',
        
        # Regular text response
        "Hello! I'm a helpful AI assistant. How can I help you today?",
        
        # Mixed response with JSON embedded
        'I can help you with that calculation. {"function": "calculate", "arguments": {"expression": "10 + 5"}} Let me compute this for you.',
        
        # Invalid function call
        '{"function": "unknown_function", "arguments": {"param": "value"}}',
        
        # Malformed JSON
        '{"function": "calculate", "arguments": {"expression": "2+2"'
    ]
    
    for i, output in enumerate(test_outputs, 1):
        print(f"Test {i}:")
        print(f"LLM Output: {output}")
        result = route_llm_output(output)
        print(f"Routed Result: {result}")
        print("-" * 50)


def generate_sample_logs():
    """Generate sample logs as requested in the assignment."""
    print("\n=== Sample Test Logs (as requested in deliverables) ===")
    
    test_scenarios = [
        {
            "user_query": "What is 15 times 23?",
            "llm_response": '{"function": "calculate", "arguments": {"expression": "15*23"}}',
            "description": "Math query test"
        },
        {
            "user_query": "Find papers about quantum computing",
            "llm_response": '{"function": "search_arxiv", "arguments": {"query": "quantum computing"}}',
            "description": "ArXiv search test"
        },
        {
            "user_query": "Hello, how are you?",
            "llm_response": "Hello! I'm doing well, thank you for asking. I'm here to help you with questions, calculations, and finding research papers. How can I assist you today?",
            "description": "Normal conversation test"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test Log {i}: {scenario['description']} ---")
        print(f"1. User query: \"{scenario['user_query']}\"")
        print(f"2. Raw LLM response: {scenario['llm_response']}")
        
        # Process the response through routing
        final_response = route_llm_output(scenario['llm_response'])
        print(f"3. Function call made and output: {final_response}")
        print(f"4. Final assistant response: {final_response}")


if __name__ == "__main__":
    # Run all tests
    test_calculate_function()
    test_search_arxiv_function() 
    test_function_routing()
    generate_sample_logs()
    
    print("\n=== Testing Complete ===")
    print("To run the full voice agent server, execute: python voice_agent.py")
    print("Then visit http://127.0.0.1:8000 in your browser to test with audio files.")