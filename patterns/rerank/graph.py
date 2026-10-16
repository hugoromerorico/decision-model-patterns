"""Rerank: a decision model reorders retrieved documents by relevance to the query."""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_typesafe import Score, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph

load_dotenv()

TOP_K = 5

# Stand-in for a vector store. Swap `retrieve` for your retriever.
CORPUS = [
    "To reset your password, open Settings > Security and choose 'Reset password'.",
    "Passwords must be at least 12 characters and include a number.",
    "Two-factor authentication can be enabled from the Security tab.",
    "Invoices are emailed on the first business day of each month.",
    "If you are locked out, support can reset your account after verifying your identity.",
    "The mobile app supports dark mode on iOS and Android.",
    "Password reset links expire after 30 minutes.",
    "Export your data as CSV from the Reports page.",
]


class State(TypedDict):
    query: str
    documents: list[str]
    reranked: list[str]


classifier = TypeSafeClassifier()

RELEVANCE = [
    "Unrelated to the query.",
    "On topic but does not answer the query.",
    "Partially answers the query.",
    "Directly answers the query.",
]


def retrieve(state: State) -> dict:
    words = set(state["query"].lower().split())
    ranked = sorted(CORPUS, key=lambda doc: -len(words & set(doc.lower().split())))
    return {"documents": ranked[:TOP_K]}


def rerank_decision_model(state: State) -> dict:
    # One request, one Score question per document, answered independently.
    questions = {
        str(i): Score(
            instructions=f"How well does this passage answer the query?\n\nPassage: {doc}",
            criteria=RELEVANCE,
        )
        for i, doc in enumerate(state["documents"])
    }
    response = classifier.invoke({"state": state["query"], "questions": questions})
    order = sorted(questions, key=lambda i: -response.scores[i].score)
    return {"reranked": [state["documents"][int(i)] for i in order]}


graph = (
    StateGraph(State)
    .add_node(retrieve)
    .add_node(rerank_decision_model)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "rerank_decision_model")
    .add_edge("rerank_decision_model", END)
    .compile()
)


if __name__ == "__main__":
    result = graph.invoke({"query": "How do I reset my password?"})
    print("Retrieved:")
    for doc in result["documents"]:
        print(f"  {doc}")
    print("Reranked:")
    for doc in result["reranked"]:
        print(f"  {doc}")
