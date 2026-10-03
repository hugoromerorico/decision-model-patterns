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

`fake_typesafe.py` is a stand-in for `langchain_typesafe.TypeSafeClassifier`, so you can build and test patterns without an API key. It answers every question with random probabilities in the same shape as the real API. Pass a `seed` to get the same answers on every run. It's a LangChain `Runnable`, so `invoke`, `batch` and `|` all work.

```python
from fake_typesafe import Choice, FakeTypeSafeClassifier, Noul, Score

classifier = FakeTypeSafeClassifier(seed=7)

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

response.nouls["urgent"].noul  # 0.324, probability of yes
response.choices["team"].choice  # "billing", the most likely option (0.812)
response.scores["severity"].score  # 1.3, the weighted average of levels
```

The rules follow the [API spec](https://api.typesafe.ai/openapi.json): probabilities sum to about 1, `choice` is the most likely option, `score` is the probability-weighted average of the levels, and `confidence` is the top probability.
