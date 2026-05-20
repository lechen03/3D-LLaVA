"""
Generate scanrefer_train_3d_llava_with_click.json from scanrefer_train_3d_llava.json.

Injects <loc> token into refer_seg prompts to match the eval template used in
model_scanrefer_with_click.py:
  Original:  "<pc>\\n Please output the segmentation mask according to the following description. \\n{desc}"
  With click: "<pc>\\n Please output the segmentation mask of this object <loc> according to the following description. \\n{desc}"
"""
import json
import argparse
import os


OLD_PHRASE = "Please output the segmentation mask according to the following description."
NEW_PHRASE = "Please output the segmentation mask of this object <loc> according to the following description."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="playground/data/train_info/scanrefer_train_3d_llava.json")
    parser.add_argument("--output", type=str,
                        default="playground/data/train_info/scanrefer_train_3d_llava_with_click.json")
    args = parser.parse_args()

    data = json.load(open(args.input))

    count = 0
    for item in data:
        for conv in item["conversations"]:
            if conv["from"] == "human" and OLD_PHRASE in conv["value"]:
                conv["value"] = conv["value"].replace(OLD_PHRASE, NEW_PHRASE)
                count += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Processed {len(data)} samples, modified {count} prompts")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
