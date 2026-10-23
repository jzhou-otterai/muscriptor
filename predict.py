"""Replicate (cog) entry point: audio file in, MIDI file out."""

import os

from cog import BasePredictor, Input, Path

from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    resolve_instrument_names,
)
from muscriptor.transcription_model import TranscriptionModel

# Baked into the image by cog (config.json sits next to it, so no HF access
# and no gated-repo token is needed at runtime). Written by
# scripts/convert_weights_fp16.py before `cog push`: half the size of the
# float32 original, which .dockerignore keeps out of the image, since every
# cold start (billed on a private model) pulls the whole image.
WEIGHTS = "weights/model.fp16.safetensors"


class Predictor(BasePredictor):
    def setup(self):
        # Decoding is memory-bandwidth bound, and with float32 weights CUDA
        # already ran the matmuls in fp16 under autocast, so float16 weights
        # halve the traffic for near-identical numerics. The conditioners stay
        # in float32 (load_model handles that). MUSCRIPTOR_DTYPE=float32 brings
        # back the old behaviour for comparisons.
        dtype = os.environ.get("MUSCRIPTOR_DTYPE", "float16")
        self.model = TranscriptionModel.load_model(WEIGHTS, device="cuda", dtype=dtype)

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe (wav, mp3, flac, …)"),
        more_audio: list[Path] = Input(
            default=None,
            description=(
                "More recordings to transcribe in the same call. They are decoded together "
                "with `audio`: chunk k of every recording shares one batch, and each "
                "recording keeps its own prelude forcing, so the result matches "
                "transcribing them one at a time while the GPU works on all of them at "
                "each step. Recordings of similar length batch best."
            ),
        ),
        instruments: str = Input(
            default="",
            description=(
                "Leave empty (recommended) to let the model detect the instruments. "
                "Otherwise, comma-separated instrument groups actually playing in the "
                "recording. The list does two things: it tells the model these "
                "instruments are present, and it forbids every other instrument. A long "
                "list therefore makes transcriptions slow and over-dense; it is not a way "
                "to exclude a few instruments. Case-insensitive, unambiguous abbreviations "
                "are accepted ('piano' is ambiguous: use acoustic_piano or electric_piano). "
                "Names: " + ", ".join(MT3_FULL_PLUS_GROUP_NAMES)
            ),
        ),
        batch_size: int = Input(
            default=1,
            ge=1,
            le=16,
            description=(
                "5-second chunks of one recording decoded at once. Above 1 requires "
                "prelude_forcing=false, which changes the transcription; to batch without "
                "that, send several recordings with more_audio instead."
            ),
        ),
        prelude_forcing: bool = Input(
            default=True,
            description=(
                "Start each chunk from the notes still sounding at the end of the previous "
                "one. Better at chunk boundaries, but chunks then run strictly one at a time."
            ),
        ),
        detect_tempo: bool = Input(
            default=True,
            description=(
                "Fit a constant tempo to write bars and tempo into the MIDI (falls back to "
                "120 BPM when the fit fails). Off skips the beat tracker."
            ),
        ),
    ) -> list[Path]:
        """One MIDI per recording: `audio` first, then `more_audio` in order."""
        if batch_size > 1 and prelude_forcing:
            raise ValueError("batch_size above 1 requires prelude_forcing=false")
        if batch_size > 1 and more_audio:
            raise ValueError("batch_size batches one recording; with more_audio, recordings are batched together")
        names = [n for n in instruments.split(",") if n.strip()]
        options = dict(
            instruments=resolve_instrument_names(names) or None,
            prelude_forcing=prelude_forcing,
            detect_tempo="best-effort" if detect_tempo else False,
            # Chord markers stay off: the image doesn't install librosa (the chord
            # model's input feature), and the MIDI keeps to MuScriptor's own notes.
            recognize_chords=False,
        )
        if more_audio:
            results = self.model.transcribe_many_and_postprocess(
                [str(audio)] + [str(a) for a in more_audio], **options
            )
        else:
            results = [self.model.transcribe_and_postprocess(str(audio), batch_size=batch_size, **options)]
        outputs = []
        for i, (midi, _grid) in enumerate(results):
            out = Path(f"/tmp/transcription_{i}.mid")
            out.write_bytes(midi)
            outputs.append(out)
        return outputs
