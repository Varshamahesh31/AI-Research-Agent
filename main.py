import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools import tools


load_dotenv()


# ---------------------------------------
# Structured response
# ---------------------------------------

class ResearchResponse(BaseModel):

    topic: str = Field(
        description="The main topic of the user's query."
    )

    summary: str = Field(
        description="A clear summary of the research."
    )

    tool_used: list[str] = Field(
        description="The tools used during the research."
    )


# ---------------------------------------
# Gemini
# ---------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ---------------------------------------
# Agent
# ---------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ResearchResponse,
    system_prompt="""
You are a research assistant.

Your job is to research topics and provide useful answers.

Rules:

1. Understand the user's question.
2. Use web_search when recent information is required.
3. Use Wikipedia when general background information is useful.
4. Use save_to_txt when the user asks to save research.
5. Return the final answer using the ResearchResponse structure.
6. Keep the summary clear and concise.
7. List the tools actually used in tool_used.
""",
)


# ---------------------------------------
# User input
# ---------------------------------------

query = input("What can I help you with? ")


# ---------------------------------------
# Run agent
# ---------------------------------------

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


# ---------------------------------------
# Get structured response
# ---------------------------------------

structured_response = response["structured_response"]


# ---------------------------------------
# Print
# ---------------------------------------

print("\n========== RESEARCH RESULT ==========\n")

print("Topic:")
print(structured_response.topic)

print("\nSummary:")
print(structured_response.summary)

print("\nTools Used:")
print(structured_response.tool_used)