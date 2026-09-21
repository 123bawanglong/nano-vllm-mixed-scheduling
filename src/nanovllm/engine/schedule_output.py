from dataclasses import dataclass, field

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class ScheduleOutput:
    decode_seqs: tuple[Sequence, ...]
    prefill_seqs: tuple[Sequence, ...]
    num_decode_tokens: int = field(init=False)
    num_prefill_tokens: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'num_decode_tokens', len(self.decode_seqs))
        object.__setattr__(self, 'num_prefill_tokens', sum(s.num_scheduled_tokens for s in self.prefill_seqs))

    @property
    def num_scheduled_tokens(self):
        return self.num_decode_tokens + self.num_prefill_tokens


@dataclass(frozen=True, slots=True)
class StepStats:
    num_prefill_tokens: int
    num_decode_tokens: int
    prefill_seconds: float
    decode_seconds: float
    elapsed_seconds: float
