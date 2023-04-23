import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizerFast
from create_model import SEQ_LEN, SPECIAL_TOKEN, BERTS
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizerFast.from_pretrained(BERTS["tiny"], max_len=SEQ_LEN)
btc = load_dataset("tner/btc")
conll2003 = load_dataset("tner/conll2003")
wikiann = load_dataset("wikiann", "en")

btc_df = pd.concat([
    btc["train"].to_pandas(),
    btc["validation"].to_pandas(),
    btc["test"].to_pandas(),
])
conll2003_df = pd.concat([
    conll2003["train"].to_pandas(),
    conll2003["validation"].to_pandas(),
    conll2003["test"].to_pandas(),
])
wikiann_df = pd.concat([
    wikiann["train"].to_pandas(),
    wikiann["test"].to_pandas(),
    wikiann["validation"].to_pandas(),
])

conll2003_tag2id = {
    "O": 0,
    "B-ORG": 1,
    "B-MISC": 2,
    "B-PER": 3,
    "I-PER": 4,
    "B-LOC": 5,
    "I-ORG": 6,
    "I-MISC": 7,
    "I-LOC": 8
}
btc_tag2id = {
    "B-LOC": 0,
    "B-ORG": 1,
    "B-PER": 2,
    "I-LOC": 3,
    "I-ORG": 4,
    "I-PER": 5,
    "O": 6
}
wikiann_tag2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
}
CONLL2003_TO_BTC_TAG = {
    0: 6,
    1: 1,
    2: 6,
    3: 2,
    4: 5,
    5: 0,
    6: 4,
    7: 6,
    8: 3
}

conll2003_df["tags"] = conll2003_df["tags"].apply(lambda tags: [CONLL2003_TO_BTC_TAG[t] for t in tags])

WIKIANN_TO_BTC_TAG = {
    0: 6,
    1: 2,
    2: 5,
    3: 1,
    4: 4,
    5: 0,
    6: 3
}

wikiann_df["tags"] = wikiann_df["ner_tags"].apply(lambda tags: [WIKIANN_TO_BTC_TAG[t] for t in tags])
del wikiann_df["ner_tags"]
del wikiann_df["langs"]
del wikiann_df["spans"]
df = pd.concat([conll2003_df, btc_df, wikiann_df])
df = df.rename(columns={"tags": "ner_tags"})

train_df, test_df = train_test_split(df, test_size=0.1)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
})


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        # Only label the first token of a given word.

        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(SPECIAL_TOKEN)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = dataset.map(tokenize_and_align_labels, batched=True)
dataset.save_to_disk("my_ner_dataset")
