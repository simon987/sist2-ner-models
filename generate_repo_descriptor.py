import json
from glob import glob
import typer
import os


def du(path):
    return sum(
        os.stat(file).st_size
        for file in glob(os.path.join(path, "*"), recursive=True)
    )


def main(models_path: str = ".", output: str = "repo.json"):
    models = []

    for model in glob(os.path.join(models_path, "**/model.json"), recursive=True):

        if "-tiny" in model:
            # Skip -tiny model (too inaccurate)
            continue

        model_dir = os.path.dirname(model)

        model_name = os.path.basename(model_dir)
        model_name = model_name.replace("-js8", "-q8")
        model_name = model_name.replace("-js16", "-q16")
        model_name = model_name.replace("-js", "")

        size = du(model_dir)

        with open(os.path.relpath(model, models_path).replace("model.json", "id2label.json")) as f:
            id2label = json.load(f)

        models.append({
            "name": model_name,
            "size": size,
            "modelPath": os.path.relpath(model, models_path),
            "vocabPath": os.path.relpath(model, models_path).replace("model.json", "vocab.json"),
            "id2label": id2label,
            "legend": {
                "LOC": "Location",
                "ORG": "Organisation",
                "PER": "Person",
            },
            "humanLabels": {
                "B-LOC": "LOC",
                "I-LOC": "LOC",
                "B-PER": "PER",
                "I-PER": "PER",
                "B-ORG": "ORG",
                "I-ORG": "OGR",
                "O": "O"
            },
            "labelStyles": {
                "LOC": {
                    "backgroundColor": "#636EFF",
                    "color": "#fff",
                    "border-radius": "3px",
                    "padding": "0 2px"
                },
                "ORG": {
                    "backgroundColor": "#ffb329",
                    "border-radius": "3px",
                    "padding": "0 2px"
                },
                "PER": {
                    "backgroundColor": "#ef0042",
                    "border-radius": "3px",
                    "padding": "0 2px"
                },
                "O": {}
            },
            "default": True if model_name == "bert-ner-medium-q8" else False
        })

    models = list(sorted(models, key=lambda m: m["size"]))

    with open(output, "w") as f:
        json.dump(models, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
