import argparse
import logging
import json

from brainsoft_code_challenge.scraping import scrape_all


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-api-token", type=str, help="GitHub API token")
    parser.add_argument("--output-path", type=str, default="scraped_docs.json", help="Path to save the results")
    args = parser.parse_args()

    results = scrape_all(args.github_api_token)
    logging.info(f"Successfully obtained {len(results)} documents")
    with open(args.output_path, "w") as f:
        f.write(json.dumps(results, ensure_ascii=False))
