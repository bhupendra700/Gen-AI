from os import getenv
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=getenv("API_KEY"))

print("Sending request...", flush=True)

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with only OK.",
        config={"http_options": {"timeout": 30000}},
    )

    print("Response:", response.text, flush=True)

except Exception as e:
    print("Error:", repr(e), flush=True)
