"""Write a float16 copy of the Replicate image's weights: weights/model.fp16.safetensors.

The large model's float32 file is 5.5 GB, and a private Replicate model reads all of it on every (billed)
cold start. predict.py loads the model in float16 anyway (load_model rounds every weight to float16, then
returns the small conditioners to float32), so storing it that way halves the file and changes nothing that
runs. The six conditioner tensors stay float32 in the file: they are tiny, and that keeps the file right if
load_model ever stops rounding them.

  python scripts/convert_weights_fp16.py [weights/model.safetensors] [weights/model.fp16.safetensors]
"""

import sys

from safetensors.torch import load_file, save_file

KEEP_FP32 = ("condition_provider.",)


def convert(src: str, dst: str) -> None:
    state = load_file(src)
    out = {
        k: v if (not v.is_floating_point() or k.startswith(KEEP_FP32)) else v.half()
        for k, v in state.items()
    }
    save_file(out, dst)
    kept = sum(1 for k, v in out.items() if v.is_floating_point() and v.dtype.itemsize == 4)
    print(f"{dst}: {len(out)} tensors, {kept} kept in float32")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "weights/model.safetensors"
    dst = sys.argv[2] if len(sys.argv) > 2 else "weights/model.fp16.safetensors"
    convert(src, dst)
