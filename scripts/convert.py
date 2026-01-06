#!/usr/bin/env python3
import argparse
import inspect
import os
from rknn.api import RKNN


def call_supported(fn, **kwargs):
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return fn(**filtered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="rk3588", choices=["rk3588", "rk3568", "rk3576"])
    ap.add_argument("--opt", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--dataset", default=None, help="txt file for quant dataset when --int8")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        raise SystemExit("ONNX not found: " + args.onnx)

    if args.int8 and not args.dataset:
        raise SystemExit("--int8 requires --dataset <dataset.txt>")

    rknn = RKNN(verbose=args.verbose)

    call_supported(
        rknn.config,
        target_platform=args.target,
        optimization_level=args.opt,
        float_dtype="float16" if args.fp16 else "float32",
        quantized_dtype="asymmetric_quantized-u8" if args.int8 else None,
        quantized_algorithm="normal" if args.int8 else None,
        single_core_mode=False,
    )

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")

    ret = call_supported(
        rknn.build,
        do_quantization=bool(args.int8),
        dataset=args.dataset if args.int8 else None,
        rknn_batch_size=None,
    )
    if ret not in (None, 0):
        raise SystemExit(f"build failed: {ret}")

    ret = rknn.export_rknn(args.out)
    if ret != 0:
        raise SystemExit(f"export_rknn failed: {ret}")

    print("OK:", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
