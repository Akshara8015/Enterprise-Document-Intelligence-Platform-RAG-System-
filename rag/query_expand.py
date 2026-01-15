from typing import List
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="phi3",
    temperature=0
)

QUERY_EXPANSION_PROMPT = """
You are expanding a search query for legal document retrieval.

Given a user query, generate 3 alternative search queries
that use different wording but preserve the same meaning.

Rules:
- Keep queries short
- Do NOT answer the question
- Do NOT add explanations

User Query:
{query}

Expanded Queries:
"""

prompt = PromptTemplate(
    template=QUERY_EXPANSION_PROMPT,
    input_variables=["query"]
)

def expand_query_llm(query: str, max_expansions: int = 3) -> List[str]:
    try:
        response = llm.invoke(prompt.format(query=query))
        lines = response.content.split("\n")

        expansions = [
            q.strip("-• ").strip() for q in lines if len(q.strip()) > 5
        ]

        return [query] + expansions[:max_expansions]

    except Exception:
        # Fallback to original query
        return [query]


def expand_query_fallback(query: str) -> List[str]:
    keywords = {
        "termination": [
            "termination clause",
            "grounds for termination",
            "termination conditions"
        ],
        "employment": [
            "employment agreement",
            "employee contract"
        ],
        "notice": [
            "notice period",
            "prior notice"
        ]
    }

    expansions = [query]

    for word, variants in keywords.items():
        if word in query.lower():
            expansions.extend(variants)

    return list(set(expansions))[:4]


def expand_query(query: str) -> List[str]:
    expansions = expand_query_llm(query)

    # If LLM fails or returns only original query
    if len(expansions) <= 1:
        expansions = expand_query_fallback(query)

    return list(dict.fromkeys(expansions))  # preserve order, remove duplicates
