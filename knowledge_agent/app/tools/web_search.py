from groq import Groq
import os

class WebSearchTool:
    def __init__(self):
        self.websearch_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def websearch(self, query: str) -> str:
        print("websearch function is running")
        response = self.websearch_client.chat.completions.create(
            model="compound-beta",
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return response.choices[0].message.content
