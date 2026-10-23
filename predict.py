"""Replicate (cog) entry point: audio file in, MIDI file out."""

from cog import BasePredictor, Input, Path

from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    resolve_instrument_names,
)
from muscriptor.transcription_model import TranscriptionModel

# Baked into the image by cog (config.json sits next to it, so no HF access
# and no gated-repo token is needed at runtime).
WEIGHTS = "weights/model.safetensors"


class Predictor(BasePredictor):
    def setup(self):
        self.model = TranscriptionModel.load_model(WEIGHTS, device="cuda")

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe (wav, mp3, flac, …)"),
        instruments: str = Input(
            default="",
            description=(
                "Comma-separated instrument groups to restrict the transcription "
                "to; anything else is forbidden from being decoded. Leave empty to "
                "let the model decide. Case-insensitive, unambiguous abbreviations "
                "are accepted (e.g. 'timp,cello,dist'). Names: "
                + ", ".join(MT3_FULL_PLUS_GROUP_NAMES)
            ),
        ),
    ) -> Path:
        # ponytail: no sampling/tempo knobs — defaults match the CLI; add Inputs
        # for them when someone actually asks.
        names = [n for n in instruments.split(",") if n.strip()]
        out = Path("/tmp/transcription.mid")
        out.write_bytes(
            self.model.transcribe_to_midi(
                str(audio),
                instruments=resolve_instrument_names(names) or None,
            )
        )
        return out
