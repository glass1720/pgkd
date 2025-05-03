import os
from abc import ABC, abstractmethod


class Teacher(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class OpenAITeacher(Teacher):
    def __init__(self, model="gpt-4o"):
        import openai

        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, prompt):
        return (
            self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            .choices[0]
            .message.content
        )


class OpenAIJSONTeacher(OpenAITeacher):
    """Uses OpenAI JSON mode to *guarantee* the model returns a JSON object."""

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class ClaudeTeacher(Teacher):
    def __init__(self, model="claude-3-sonnet-20240229"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def generate(self, prompt):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        return msg.content[0].text


class GeminiTeacher(Teacher):
    def __init__(self, model="gemini-1.5-pro-preview"):
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt):
        return self.model.generate_content(
            prompt, generation_config={"response_mime_type": "application/json"}
        ).text


PROVIDERS = {
    "OpenAI GPT-4o": OpenAITeacher,
    "Anthropic Claude-3": ClaudeTeacher,
    "Google Gemini": GeminiTeacher,
    "OpenAI GPT-4o JSON": OpenAIJSONTeacher,
}
