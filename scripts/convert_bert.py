#!/usr/bin/env python3
import argparse
from rknn.api import RKNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--opt", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rknn = RKNN(verbose=args.verbose)

    seq = int(args.seq)
    mean = [0.0] * seq
    std = [1.0] * seq

    rknn.config(
        target_platform=args.target,
        optimization_level=args.opt,
        mean_values=[mean, mean, mean],
        std_values=[std, std, std],
        quantized_dtype="w8a8",
        float_dtype="float16" if args.fp16 else "float32",
    )

    ret = rknn.load_onnx(
        model=args.onnx,
        inputs=["input_ids", "attention_mask", "token_type_ids"],
        input_size_list=[[1, seq], [1, seq], [1, seq]],
        outputs=["last_hidden_state"],
    )
    if ret != 0:
        raise SystemExit("load_onnx failed")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise SystemExit("build failed")

    ret = rknn.export_rknn(args.out)
    if ret != 0:
        raise SystemExit("export_rknn failed")

    rknn.release()
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
