import typer
import json

from transformers import BertTokenizerFast

from create_model import BERTS, ID2LABEL, SEQ_LEN


def main(model_path: str):
    tokenizer = BertTokenizerFast.from_pretrained(BERTS["tiny"], max_len=SEQ_LEN)

    vocab_path = tokenizer.save_vocabulary("/tmp/")[0]
    with open(vocab_path) as f:
        vocab = [x.strip() for x in f]
    with open(f"{model_path}/vocab.json", "w") as f:
        json.dump(vocab, f)

    with open(f"{model_path}/id2label.json", "w") as f:
        json.dump(
            [v for k, v in sorted(ID2LABEL.items(), key=lambda x: x[0])],
            f
        )


if __name__ == "__main__":
    typer.run(main)
