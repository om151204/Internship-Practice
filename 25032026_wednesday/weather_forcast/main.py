import os
import requests
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

@tool(
    "WeatherTool",
    description="This tool returns the current weather conditions of input city."
)
def weather(city: str):
    resp = requests.get(f"https://wttr.in/{city}?format=j1")
    return resp.json()

model = ChatGroq(
    api_key = GROQ_API_KEY,
    model = GROQ_MODEL,
    temperature = 0.2
)

user_input = input("Enter City Name : ")
messages = [
    {
        "role": "system",
        "content": """
                      You are an helpful assistant. You only have access to the 'WeatherTool' tool.
                      Don't attempt to use any other tools like search.
                   """
    },
    {
        "role": "user",
        "content": f"Current Weather Situation of {user_input}."
    }
]

llm = model.bind_tools(
    [weather],
    tool_choice = "auto",
)

response = llm.invoke(messages)
# print(response)

messages.append(response)
for tool_calls in response.tool_calls:
    tool_result = weather.invoke(tool_calls["args"])
    messages.append(
        ToolMessage(
        tool_call_id=tool_calls["id"],
        content=str(tool_result)
        )
    )

output = llm.invoke(messages)
print(output.content)