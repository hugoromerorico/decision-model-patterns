"""A scripted fake of `langchain_typesafe.TypeSafeClassifier`, in the spirit of
`GenericFakeChatModel`: each `invoke` consumes the next scripted answer."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig


@dataclass
class Noul:
    instructions: str


@dataclass
class Choice:
    instructions: str
    criteria: dict[str, str]


@dataclass
class Score:
    instructions: str
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
    legend: list[str]
    probabilities: list[float]
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
    model: str = "fake-jev"
    usage: Usage = field(default_factory=Usage)
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")


class FakeTypeSafeClassifier(Runnable[dict[str, Any], DecisionResponse]):
    """Scripted answers are dicts of question id -> value:

    - Noul: a float, the probability of yes
    - Choice: the chosen criteria key
    - Score: the index of the chosen level in `criteria`

    A full `NoulAnswer`, `ChoiceAnswer` or `ScoreAnswer` is passed through as is.
    """

    def __init__(self, answers: Iterator[dict[str, Any]]) -> None:
        self.answers = answers

    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> DecisionResponse:
        scripted = next(self.answers)
        response = DecisionResponse()
        for qid, question in input["questions"].items():
            value = scripted[qid]
            match question:
                case Noul():
                    response.nouls[qid] = _noul(value)
                case Choice():
                    response.choices[qid] = _choice(question, value)
                case Score():
                    response.scores[qid] = _score(question, value)
        return response


def _noul(value: float | NoulAnswer) -> NoulAnswer:
    return value if isinstance(value, NoulAnswer) else NoulAnswer(noul=value)


def _choice(question: Choice, value: str | ChoiceAnswer) -> ChoiceAnswer:
    if isinstance(value, ChoiceAnswer):
        return value
    probabilities = {key: float(key == value) for key in question.criteria}
    return ChoiceAnswer(choice=value, probabilities=probabilities, confidence=1.0)


def _score(question: Score, value: int | ScoreAnswer) -> ScoreAnswer:
    if isinstance(value, ScoreAnswer):
        return value
    probabilities = [float(i == value) for i in range(len(question.criteria))]
    return ScoreAnswer(
        score=float(value),
        legend=list(question.criteria),
        probabilities=probabilities,
        confidence=1.0,
    )
