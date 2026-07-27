from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"user","content":"Hello"}
    ]
)

print(response.choices[0].message.content)

# python ./test_openai