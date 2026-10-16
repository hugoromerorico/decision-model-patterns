# Decision Model Patterns

Patterns for using **decision models** inside [LangGraph](https://langchain-ai.github.io/langgraph/) agents.

A decision model answers typed questions about some input and returns probabilities instead of text:

- **Noul**: a yes/no question. Returns the probability of yes.
- **Choice**: picks one option. Returns the chosen option, the full distribution and a confidence.
- **Score**: grades against an ordered rubric. Returns an expected value and a confidence.

When an LLM call exists only to produce a decision (route here, flag this, rate that), a decision model is faster, cheaper and returns typed output you can branch on directly. In every pattern here, the decision model is a **node** in the graph, called through [`langchain-typesafe`](https://pypi.org/project/langchain-typesafe/).

## Patterns

| Pattern | Status |
|---|---|
| [Router](#router) | ✅ |
| Evals | Planned |
| Rerank | Planned |
| Trigger | Planned |
| Tool-call protection | Planned |
| Email triage | Planned |
| Labeling | Planned |
| Guardrailing | Planned |
| LLM security | Planned |

### Router

[`patterns/router/graph.py`](patterns/router/graph.py)

<!-- Add the LangGraph Studio screenshot here: ![Router graph](patterns/router/graph.png) -->

The `decide` node asks one `Choice` question: how urgent is the request (`high`, `mid` or `low`)? It then jumps to the matching branch with `Command(goto=...)`. No prompt and no output parsing are involved, because the answer is already one of the branch names.

## Getting started

The examples run against [Decision Sandbox](https://decision-sandbox.gutan.dev), a free mock of the TypeSafe API. It returns **synthetic, random answers**, so you can explore the patterns without a model or a paid key. Expect the routing to look arbitrary.

Requires [uv](https://docs.astral.sh/uv/). [Task](https://taskfile.dev/) is optional.

```sh
cp .env.example .env
uv sync

task router   # or: uv run patterns/router/graph.py
task dev      # or: uv run langgraph dev   (opens LangGraph Studio)
```

To use a real model, point `.env` at TypeSafe:

```sh
TYPESAFE_BASE_URL=https://api.typesafe.ai
TYPESAFE_API_KEY=<your key>
```
