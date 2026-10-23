"""Tests for decoding several recordings together while each keeps its prelude forcing.

- LMModel.generate already forces any prompt position that isn't the ungenerated token, so prompts of
  different lengths can share a batch by padding the shorter rows with it; checked with a tiny random
  model: every row of a ragged batch comes out as it would alone.
- _generate_token_streams gives each recording its own tie prologue and its own stream; checked with a
  fake generate against _generate_token_stream at batch_size 1, one recording at a time.
"""

from types import SimpleNamespace

import pytest
import torch

from muscriptor.events import ChunkBoundary, ProgressEvent
from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningProvider
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import NoteEvent
from muscriptor.transcription_model import TranscriptionModel
from tests.encode_helpers import encode_index_map, encode_note_events

_MAX_SHIFT_STEPS = 1001
_INDEX = encode_index_map(_MAX_SHIFT_STEPS)


def _tok(type_: str, value: int = 0) -> int:
    return _INDEX[(type_, value)]


_EOS = _tok("EOS")


def _on(time: float, program: int, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=False, program=program, time=time, velocity=1, pitch=pitch)


def _encode(note_events, start_time=0.0) -> list[int]:
    return encode_note_events(note_events, max_shift_steps=_MAX_SHIFT_STEPS, start_time=start_time)


@pytest.fixture(scope="module")
def tokenizer() -> MT3Tokenizer:
    return MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


# ---------------------------------------------------------------------------
# LMModel.generate with prompts of different lengths (tiny random model, CPU)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model() -> LMModel:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = LMModel(
        condition_provider=ConditioningProvider(conditioners={}, device=device),
        card=16,
        dim=16,
        num_heads=2,
        hidden_scale=2,
        num_layers=1,
        max_period=10000,
        device=device,
    )
    model.eval()
    return model


def _rows(steps) -> list[list[int]]:
    return torch.stack(list(steps), dim=1).tolist()


def test_ragged_prompts_decode_each_row_as_it_would_alone(tiny_model):
    prompts = [[5, 3, 9], [7], []]
    pad = tiny_model.ungenerated_token_id
    batch = torch.full((3, 3), pad, dtype=torch.long)
    for j, p in enumerate(prompts):
        batch[j, : len(p)] = torch.tensor(p, dtype=torch.long)
    together = _rows(tiny_model.generate(prompt=batch, max_gen_len=10, use_sampling=False))

    for j, p in enumerate(prompts):
        alone = _rows(
            tiny_model.generate(
                prompt=torch.tensor([p]) if p else None,
                max_gen_len=10,
                num_samples=1,
                use_sampling=False,
            )
        )[0]
        assert together[j][: len(p)] == p
        assert together[j] == alone


# ---------------------------------------------------------------------------
# _generate_token_streams: per-recording prologues and streams (fake model)
# ---------------------------------------------------------------------------

_UNGENERATED = -1


def _fake_model(tokenizer, scripts, prompts_seen):
    """A model whose generate behaves like LMModel.generate: each row first yields its own prompt
    (forced), then its scripted tokens for that call, then EOS. ``scripts[c]`` holds one token list
    per row for the c-th generate call."""

    def generate(prompt=None, conditions=(), **kwargs):
        call = len(prompts_seen)
        rows = len(conditions)
        forced = [[] for _ in range(rows)]
        if prompt is not None:
            forced = [[t for t in row if t != _UNGENERATED] for row in prompt.tolist()]
        prompts_seen.append(forced)
        seqs = [forced[j] + scripts[call][j] for j in range(rows)]
        for t in range(max(len(s) for s in seqs) + 1):
            yield torch.tensor([s[t] if t < len(s) else tokenizer.eos_id for s in seqs])

    return SimpleNamespace(
        _model=SimpleNamespace(generate=generate, ungenerated_token_id=_UNGENERATED),
        _tokenizer=tokenizer,
        _device=torch.device("cpu"),
    )


def _single_song_stream(tokenizer, chunk_scripts, seek_times):
    """What _generate_token_stream (batch_size 1, forcing on) streams for one recording."""
    prompts_seen = []
    fake = _fake_model(tokenizer, [[s] for s in chunk_scripts], prompts_seen)
    stream = TranscriptionModel._generate_token_stream(
        fake,
        [object()] * len(seek_times),
        seek_times,
        1,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        no_eos_is_ok=True,
        prelude_forcing=True,
    )
    return [x for x in stream if not isinstance(x, ProgressEvent)]


def test_each_recording_keeps_its_own_prelude_and_stream(tokenizer):
    # Recording A: a note on program 0 left open across both boundaries; three chunks.
    a = [_encode([_on(0.1, 0, 60)]) + [_EOS], [_tok("shift", 20), _EOS], [_EOS]]
    # Recording B: a note on program 5 left open into its second (last) chunk.
    b = [_encode([_on(0.2, 5, 70)]) + [_EOS], [_tok("shift", 30), _EOS]]
    seeks_a, seeks_b = [0.0, 5.0, 10.0], [0.0, 5.0]

    def without_eos(chunk):
        return [t for t in chunk if t != _EOS]

    # The batched call gets chunk k of each recording that still has one
    scripts = [
        [without_eos(a[0]), without_eos(b[0])],
        [without_eos(a[1]), without_eos(b[1])],
        [without_eos(a[2])],
    ]
    prompts_seen = []
    fake = _fake_model(tokenizer, scripts, prompts_seen)
    streams = TranscriptionModel._generate_token_streams(
        fake,
        [[object()] * len(seeks_a), [object()] * len(seeks_b)],
        [seeks_a, seeks_b],
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        no_eos_is_ok=True,
        prelude_forcing=True,
    )

    # Each row was forced with its own recording's open notes; B has no third chunk
    assert prompts_seen[0] == [[], []]
    assert prompts_seen[1] == [tokenizer.tie_section_token_ids([(0, 60)]),
                               tokenizer.tie_section_token_ids([(5, 70)])]
    assert prompts_seen[2] == [tokenizer.tie_section_token_ids([(0, 60)])]

    # And each stream is what that recording gets decoded alone
    assert streams[0] == _single_song_stream(tokenizer, [without_eos(c) for c in a], seeks_a)
    assert streams[1] == _single_song_stream(tokenizer, [without_eos(c) for c in b], seeks_b)
    assert streams[1][0] == ChunkBoundary(0.0, 5.0)


def test_forcing_off_prompts_nothing(tokenizer):
    prompts_seen = []
    fake = _fake_model(tokenizer, [[[1], [2]], [[3], [4]]], prompts_seen)
    TranscriptionModel._generate_token_streams(
        fake,
        [[object()] * 2, [object()] * 2],
        [[0.0, 5.0], [0.0, 5.0]],
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        no_eos_is_ok=True,
        prelude_forcing=False,
    )
    assert prompts_seen == [[[], []], [[], []]]
