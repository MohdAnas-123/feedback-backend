from google import genai
from typing import Dict, Any

class Gemini:
    def __init__(self, api_key='api_key', id='gemini-2.5-flash', temperature=0.2, **kwargs):
        self.api_key = api_key
        self.id = id
        
        # New SDK semantics
        self.client = genai.Client(api_key=self.api_key)
        
        self.config = {
            "temperature": temperature,
            **kwargs
        }

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model=self.id,
            contents=prompt,
            config=self.config
        )
        return GeminiResponse(response)

class GeminiResponse:
    def __init__(self, response):
        self.raw_response = response

    @property
    def text(self) -> str:
        try:
            if hasattr(self.raw_response, "text"):
                return self.raw_response.text
            return str(self.raw_response)
        except Exception as e:
            return f"Error extracting text from response: {str(e)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
        }
