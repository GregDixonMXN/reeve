#!/usr/bin/env python3
"""
Spawn Sub-Agent Tool

This script interfaces with a local Ollama API to generate responses.

Usage:
    python spawn_sub_agent.py "<prompt>" "<model_name>"

Example:
    python spawn_sub_agent.py "Why is the sky blue?" "llama2"

Requirements:
    - requests library (pip install requests)
    - Local Ollama API running at http://localhost:11434
"""

import sys
import json
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def spawn_sub_agent(prompt, model):
    """
    Send a prompt to the Ollama API and return the complete response.
    
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model name (e.g., 'llama2', 'mistral', etc.)
    
    Returns:
        str: The complete response text from the model
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # Ollama returns newline-delimited JSON for streaming responses
        complete_response = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if "response" in json_response:
                        complete_response += json_response["response"]
                    
                    # Check if generation is complete
                    if json_response.get("done", False):
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}", file=sys.stderr)
                    continue
        
        return complete_response
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_API_URL}", file=sys.stderr)
        print("Make sure Ollama is running (e.g., 'ollama serve')", file=sys.stderr)
        sys.exit(1)
        
    except requests.exceptions.Timeout:
        print("Error: Request to Ollama API timed out", file=sys.stderr)
        sys.exit(1)
        
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error from Ollama API: {e}", file=sys.stderr)
        if response.status_code == 404:
            print(f"Model '{model}' may not be available. Check installed models with 'ollama list'", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python spawn_sub_agent.py \"<prompt>\" \"<model_name>\"", file=sys.stderr)
        print("\nExample: python spawn_sub_agent.py \"Why is the sky blue?\" \"llama2\"", file=sys.stderr)
        sys.exit(1)
    
    prompt = sys.argv[1]
    model = sys.argv[2]
    
    if not prompt.strip():
        print("Error: Prompt cannot be empty", file=sys.stderr)
        sys.exit(1)
    
    if not model.strip():
        print("Error: Model name cannot be empty", file=sys.stderr)
        sys.exit(1)
    
    # Generate response
    response_text = spawn_sub_agent(prompt, model)
    
    # Print the complete response
    print(response_text)

if __name__ == "__main__":
    main()
