# Decision Model Patterns

System one can be useful

Patterns that can work:
# evals, rerank rag, confidence threshold, tool-call-protection, email-triage, labeling, guardrailing pre-LLM, jailbreak-protection
[X] Router
[] Evals
[] Rerank
[] Trigger
[] Tool-call-protection
[] Email triage
[] Labeling
[] Guardrailing
[] LLM Security

## Fake System-1 classifier

`fake_typesafe.py` is a stand-in for `langchain_typesafe.TypeSafeClassifier`, so you can build and test patterns without an API key. It works like `GenericFakeChatModel`: you script the answers, and each `invoke` returns the next one. It's a LangChain `Runnable`, so `invoke`, `batch` and `|` all work.

```python
from fake_typesafe import Choice, FakeTypeSafeClassifier, Noul, Score

classifier = FakeTypeSafeClassifier(
    answers=iter(
        [
            {"urgent": 0.92, "team": "infra", "severity": 2},
        ]
    )
)

response = classifier.invoke(
    {
        "state": "The deploy failed twice and customers are seeing 500s.",
        "questions": {
            "urgent": Noul(instructions="Does this need attention right now?"),
            "team": Choice(
                instructions="Which team should pick this up?",
                criteria={"infra": "Deploys and incidents.", "billing": "Payments."},
            ),
            "severity": Score(
                instructions="How severe is the impact?",
                criteria=["Cosmetic.", "Degraded.", "Full outage."],
            ),
        },
    }
)

response.nouls["urgent"].noul  # 0.92
response.choices["team"].choice  # "infra"
response.scores["severity"].score  # 2.0
```
