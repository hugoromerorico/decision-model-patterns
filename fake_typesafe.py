"""A fake of `langchain_typesafe.TypeSafeClassifier` that answers every question
with random, well-formed probabilities."""

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig


@dataclass(kw_only=True)
class NoulCriteria:
    true: str | None = None
    false: str | None = None


@dataclass(kw_only=True)
class Noul:
    instructions: str | None = None
    criteria: NoulCriteria | None = None


@dataclass(kw_only=True)
class Choice:
    instructions: str | None = None
    criteria: dict[str, str | None]


@dataclass(kw_only=True)
class Score:
    instructions: str | None = None
    criteria: list[str]


@dataclass
class NoulAnswer:
    noul: float


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class DecisionResponse:
    nouls: dict[str, NoulAnswer] = field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = field(default_factory=dict)
    model: str = "jev-latest"
    usage: Usage = field(default_factory=Usage)
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")


class FakeTypeSafeClassifier(Runnable[dict[str, Any], DecisionResponse]):
    """Pass a `seed` to get the same answers on every run."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> DecisionResponse:
        response = DecisionResponse()
        for qid, question in input["questions"].items():
            match question:
                case Noul():
                    response.nouls[qid] = NoulAnswer(noul=round(self.rng.random(), 3))
                case Choice():
                    response.choices[qid] = self._choice(question)
                case Score():
                    response.scores[qid] = self._score(question)
        return response

    def _probabilities(self, keys: list[str]) -> dict[str, float]:
        weights = [self.rng.random() for _ in keys]
        total = sum(weights)
        return {key: round(w / total, 3) for key, w in zip(keys, weights)}

    def _choice(self, question: Choice) -> ChoiceAnswer:
        probabilities = self._probabilities(list(question.criteria))
        choice = max(probabilities, key=probabilities.__getitem__)
        return ChoiceAnswer(
            choice=choice,
            probabilities=probabilities,
            confidence=probabilities[choice],
        )

    def _score(self, question: Score) -> ScoreAnswer:
        legend = {str(i): level for i, level in enumerate(question.criteria)}
        probabilities = self._probabilities(list(legend))
        return ScoreAnswer(
            score=round(sum(int(k) * p for k, p in probabilities.items()), 3),
            legend=legend,
            probabilities=probabilities,
            confidence=max(probabilities.values()),
        )
