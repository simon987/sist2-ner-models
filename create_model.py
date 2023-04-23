import evaluate
import os
import json
import numpy as np
import typer
from datasets import load_from_disk
from transformers import BertTokenizerFast, TFBertForTokenClassification, DataCollatorForTokenClassification, \
    create_optimizer
from transformers.keras_callbacks import KerasMetricCallback

LABEL2ID = {
    "B-LOC": 0,
    "B-ORG": 1,
    "B-PER": 2,
    "I-LOC": 3,
    "I-ORG": 4,
    "I-PER": 5,
    "O": 6
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

BERTS = {
    "tiny": "google/bert_uncased_L-2_H-128_A-2",
    "mini": "google/bert_uncased_L-4_H-256_A-4",
    "small": "google/bert_uncased_L-4_H-512_A-8",
    "medium": "google/bert_uncased_L-8_H-512_A-8",
}

NUM_LABELS = len(LABEL2ID)
BATCH_SIZE = 64
SPECIAL_TOKEN = -100
SEQ_LEN = 128


def create_model(output_path: str, num_train_epochs: int = 10, base_model: str = "medium"):
    base_model = BERTS[base_model]

    tokenizer = BertTokenizerFast.from_pretrained(base_model, max_len=SEQ_LEN)

    model = TFBertForTokenClassification.from_pretrained(
        base_model, num_labels=NUM_LABELS, from_pt=True,
        id2label=ID2LABEL, label2id=LABEL2ID
    )

    dataset = load_from_disk("my_ner_dataset")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

    tf_train_set = model.prepare_tf_dataset(
        dataset["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        dataset["test"],
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p

        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != SPECIAL_TOKEN]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != SPECIAL_TOKEN]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=(len(dataset["train"]) // BATCH_SIZE) * num_train_epochs,
        weight_decay_rate=0.01,
        num_warmup_steps=0,
    )

    model.compile(optimizer=optimizer)

    hist = model.fit(x=tf_train_set, epochs=num_train_epochs, callbacks=[metric_callback])

    model.save(output_path)

    with open(os.path.join(output_path, "history.json"), "w") as f:
        json.dump(hist.history, f)

    print("Done!")


if __name__ == "__main__":
    typer.run(create_model)
