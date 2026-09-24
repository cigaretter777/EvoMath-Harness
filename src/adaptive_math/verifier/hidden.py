"""Offline-only answer evaluation boundary for reinforcement learning."""

from adaptive_math.core.types import ReferenceAnswer
from adaptive_math.verifier.service import (
    VerifierResult,
    no_final_answer_verdict,
    verify_answer,
)


class HiddenVerifier:
    """Retains a reference privately and exposes only a terminal verdict."""

    def __init__(self, reference: ReferenceAnswer, task_id: str) -> None:
        self.__reference = reference
        self.__task_id = task_id

    def evaluate(self, answer: str) -> VerifierResult:
        return verify_answer(answer, self.__reference, task_id=self.__task_id)

    def no_final_answer(self) -> VerifierResult:
        """Verdict for a terminated rollout that never submitted an answer.

        Deliberately not ``evaluate("")``: an empty string is a prediction the
        verifiers try to parse, and for expression-typed tasks that goes through
        the symbolic worker and can come back as internal_error -- a different
        fact, and one that would be logged as a verifier fault."""
        return no_final_answer_verdict()
