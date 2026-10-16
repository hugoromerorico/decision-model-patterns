"""Router: a decision model picks which branch handles each request."""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_typesafe import Choice, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

load_dotenv()

Priority = Literal["high", "mid", "low"]


class State(TypedDict):
    request: str
    priority: Priority
    result: str


classifier = TypeSafeClassifier()

PRIORITY = Choice(
    instructions="How urgently does this request need to be handled?",
    criteria={
        "high": "Outage or data loss affecting customers right now.",
        "mid": "Degraded experience that should be fixed today.",
        "low": "Cosmetic or minor issue that can wait.",
    },
)


def decide(state: State) -> Command[Priority]:
    response = classifier.invoke(
        {"state": state["request"], "questions": {"priority": PRIORITY}}
    )
    priority = response.choices["priority"].choice
    return Command(update={"priority": priority}, goto=priority)


def high(state: State) -> dict:
    return {"result": "Escalated immediately"}


def mid(state: State) -> dict:
    return {"result": "Queued for today"}


def low(state: State) -> dict:
    return {"result": "Added to backlog"}


graph = (
    StateGraph(State)
    .add_node(decide)
    .add_node(high)
    .add_node(mid)
    .add_node(low)
    .add_edge(START, "decide")
    .add_edge("high", END)
    .add_edge("mid", END)
    .add_edge("low", END)
    .compile()
)


if __name__ == "__main__":
    for request in [
        "Production database is down for all customers",
        "Export button is slow on large reports",
        "Typo in the footer",
    ]:
        result = graph.invoke({"request": request})
        print(f"[{result['priority']}] {request} -> {result['result']}")
