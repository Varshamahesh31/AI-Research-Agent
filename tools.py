from datetime import datetime

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool


# Save research to a text file


def save_to_txt(
    data: str,
    filename: str = "research_output.txt"
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    formatted_text = (
        "--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    return f"Research successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves research data to a text file."
)

# DuckDuckGo Web Search

search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the web for current and recent information. "
        "Use this tool when the user asks about people, "
        "companies, technology, current events, or any "
        "information that may require up-to-date information."
    )
)

# Tools available to the agent
tools = [
    search_tool,
    save_tool,
]