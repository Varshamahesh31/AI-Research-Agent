from datetime import datetime

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


# Web search
search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for recent information."""
    return search.run(query)


# Wikipedia
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=2000,
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper
)


# Save research to file
@tool
def save_to_txt(
    data: str,
    filename: str = "research_output.txt"
) -> str:
    """Save research data to a text file."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    formatted_text = (
        "--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    return f"Data successfully saved to {filename}"


tools = [
    web_search,
    wiki_tool,
    save_to_txt,
]