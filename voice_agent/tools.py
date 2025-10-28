import logging
from livekit.agents import function_tool, RunContext
import os
from groq import Groq
from dotenv import load_dotenv



load_dotenv()
websearch_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@function_tool()
async def search_web(context: RunContext, query: str) -> str:
    try:
        system_prompt = {
            "role": "system",
            "content": "You are a perfect assistant. Provide **short and concise answers**. Be direct, clear, and avoid unnecessary details."
        }
        response = websearch_client.chat.completions.create(
            model="compound-beta",
            messages=[
                system_prompt,
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error searching the web for '{query}': {e}")
        return f"An error occurred while searching the web for '{query}'."    
