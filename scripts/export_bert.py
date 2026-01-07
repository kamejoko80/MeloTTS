#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer, AutoModel

class BertWrap(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.eval()

    def forward(self, input_ids_i32, attention_mask_i32, token_type_ids_i32):
        input_ids = input_ids_i32.to(torch.int64)
        attention_mask = attention_mask_i32.to(torch.int64)
        token_type_ids = token_type_ids_i32.to(torch.int64)
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return out.last_hidden_state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--out", default="bert_base_uncased_S128.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    x = tok("hello world", return_tensors="pt", padding="max_length", truncation=True, max_length=args.seq)

    input_ids = x["input_ids"].to(torch.int32)
    attn = x["attention_mask"].to(torch.int32)
    tt = x.get("token_type_ids", torch.zeros_like(input_ids)).to(torch.int32)

    m = BertWrap(args.model)

    with torch.no_grad():
        torch.onnx.export(
            m,
            (input_ids, attn, tt),
            args.out,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
