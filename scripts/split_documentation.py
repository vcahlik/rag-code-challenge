from brainsoft_code_challenge.utils import load_environment

load_environment()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

from brainsoft_code_challenge.data_loading.splitting import split_document  # noqa: E402

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="scraped_docs.json", help="Path to input data")
    parser.add_argument("--output-path", type=str, default="split_docs.json", help="Path to save the results")
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    results = []
    for document in data:
        split_parts = split_document(document)
        if len(split_parts) > 1:
            logging.info(f"Split {document['source_path']} into {len(split_parts)} parts")
        results.extend(split_parts)

    with open(args.output_path, "w") as f:
        f.write(json.dumps(results, ensure_ascii=False))
