import os

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools import tools


# Load environment variables
load_dotenv()


# Check API key

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Please add it to your .env file."
    )



# Response structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    tool_used: list[str]



# Gemini model

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
)



# Create agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a research assistant.

Your job is to answer the user's questions accurately.

Rules:

1. Use the web_search tool when the question requires
   current, recent, or factual information.

2. Use the save_text_to_file tool when the user asks
   you to save research.

3. For simple greetings or casual conversation,
   answer directly without using tools.

4. Always return the final answer in this exact JSON format:

{
    "topic": "short topic name",
    "summary": "clear answer to the user's question",
    "tool_used": ["name of tool used"]
}

If no tool was used, return:

"tool_used": []

Do not add markdown or any text outside the JSON.
""",
)



# Get user input

query = input("What can I help you with? ")



# Run agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }
)



# Get final message

final_message = response["messages"][-1]

content = final_message.content


# Gemini/LangChain can sometimes return
# content as a list of blocks.
if isinstance(content, list):

    text_parts = []

    for part in content:
        if isinstance(part, dict) and "text" in part:
            text_parts.append(part["text"])
        elif isinstance(part, str):
            text_parts.append(part)

    content = "".join(text_parts)



# Display result
print("\n========== RESEARCH RESULT ==========\n")
print(content)