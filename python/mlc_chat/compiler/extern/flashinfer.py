"""FlashInfer library."""
import dataclasses


@dataclasses.dataclass
class FlashInfer:
    """A fast kernel library for LLM inference."""

    rope_scale: float = 1.0
    rope_theta: float = 10000.0

    def configure(
        self,
        rope_scale: float,
        rope_theta: float,
    ):
        """Configure FlashInfer as an external operator

        Parameters
        ----------
        rope_scale : float
            Scaling factor for the RoPE embedding.

        rope_theta : float
            The base period of the RoPE embedding.
        """
        self.rope_scale = rope_scale
        self.rope_theta = rope_theta
