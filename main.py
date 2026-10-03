from typing import Literal, cast

from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

Priority = Literal["high", "mid", "low"]


class State(TypedDict):
    request: str
    result: str


llm = GenericFakeChatModel(
    messages=iter([AIMessage(content="high"), AIMessage(content="mid"), "low"])
)


def decide(state: State) -> Command[Priority]:
    decision = llm.invoke(f"Classify the priority of this request: {state['request']}")
    return Command(goto=cast(Priority, decision.content))


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


def main():
    for request in [
        "Production database is down for all customers",
        "Export button is slow on large reports",
        "Typo in the footer",
    ]:
        result = graph.invoke({"request": request})
        print(f"{request} -> {result['result']}")


if __name__ == "__main__":
    main()
