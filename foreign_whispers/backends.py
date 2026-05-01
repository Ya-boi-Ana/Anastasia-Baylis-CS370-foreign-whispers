"""Extended TTS interface with soft duration targets."""

import abc


class DurationAwareTTSBackend(abc.ABC):
    """Abstract TTS backend that accepts optional alignment metadata."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        duration_hint_s: float | None = None,
        pause_budget_s: float | None = None,
        max_stretch_factor: float = 1.15,
    ) -> float:
        """Synthesize text to output_path and return duration in seconds."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"