from os import getenv
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = getenv("API_KEY")
client = genai.Client(api_key=api_key)

history = []

system_instruction = """
Respond only in plain text.
Do not use Markdown.
Do not use headings.
Do not use bullet points or numbered lists.
Do not use bold, italic, tables, or code blocks.
Write answers as normal sentences and paragraphs.
"""


def chat():
    while True:
        userProblem = input("You: ")

        if userProblem.lower() == "exit":
            break

        history.append(
            {"type": "user_input", "content": [{"type": "text", "text": userProblem}]}
        )

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=history,
            generation_config={"temperature": 1.0},
            system_instruction=system_instruction,
            store=False,
        )

        print("AI: ", interaction.output_text, flush=True)
        history.append(
            {
                "type": "model_output",
                "content": [{"type": "text", "text": interaction.output_text}],
            }
        )


chat()
