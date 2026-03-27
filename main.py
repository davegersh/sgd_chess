import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset
from dotenv import load_dotenv
from sdg_hub.core.flow import Flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 2
MIN_EVAL_SCORE = 5.0
PROGRESS_FILE = "progress.json"


def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def is_valid_score(eval_str: str) -> bool:
    try:
        score = float(str(eval_str).strip())
        return score >= MIN_EVAL_SCORE
    except ValueError, TypeError:
        return False


def get_score(eval_str: str) -> float:
    try:
        return float(str(eval_str).strip())
    except ValueError, TypeError:
        return 0.0


def generate_with_retry(
    flow: Flow, dataset_item, max_retries: int = MAX_RETRIES
) -> Optional[dict]:
    for attempt in range(max_retries):
        try:
            result = flow.generate(dataset_item)
            result_dict = dict(result[0]) if hasattr(result, "__getitem__") else {}
            q = str(result_dict.get("question", "")).strip()
            a = str(result_dict.get("answer", "")).strip()
            e = str(result_dict.get("eval", "")).strip()
            if q and a and e:
                return result_dict
            logger.warning(f"Attempt {attempt + 1}: Empty or malformed response")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            delay = BASE_DELAY * (2**attempt)
            logger.info(f"Retrying in {delay}s...")
            time.sleep(delay)

    logger.error(f"All {max_retries} attempts failed")
    return None


def main():
    load_dotenv()

    dataset = Dataset.from_csv("chess_openings.csv")
    flow = Flow.from_yaml("chess_flow/flow.yaml")

    flow.set_model_config(
        model="openai/llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    logger.info("Dry run...")
    dry = flow.dry_run(dataset, sample_size=1)
    logger.info(f"Dry run passed in {dry['execution_time_seconds']:.2f}s")

    progress = load_progress()
    completed = set(progress["completed"])
    failed_attempts = progress["failed"]

    output_file = "chess_openings_qa.jsonl"
    existing_entries = []
    if Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                try:
                    existing_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Loaded {len(existing_entries)} existing entries")

    filtered_entries = [e for e in existing_entries if is_valid_score(e.get("eval", 0))]
    logger.info(
        f"Filtered to {len(filtered_entries)} entries with eval >= {MIN_EVAL_SCORE}"
    )

    with open(output_file, "w") as f:
        for entry in filtered_entries:
            f.write(json.dumps(entry) + "\n")

    total = len(dataset)
    new_count = 0
    skip_count = 0

    for i in range(total):
        opening_name = dataset[i]["name"]

        if i in completed:
            logger.info(
                f"[{i + 1}/{total}] Skipping '{opening_name}' (already completed)"
            )
            skip_count += 1
            continue

        if i in failed_attempts and failed_attempts[str(i)] >= MAX_RETRIES:
            logger.info(
                f"[{i + 1}/{total}] Skipping '{opening_name}' (max retries exceeded)"
            )
            skip_count += 1
            continue

        logger.info(f"[{i + 1}/{total}] Processing '{opening_name}'...")

        result = generate_with_retry(flow, dataset.select([i]))

        if result is None:
            failed_attempts[str(i)] = failed_attempts.get(str(i), 0) + 1
            progress["failed"] = failed_attempts
            save_progress(progress)
            continue

        eval_score = get_score(result["eval"][0])

        if not is_valid_score(result["eval"][0]):
            logger.warning(f"Low quality (eval={eval_score:.1f}), skipping...")
            completed.add(i)
            progress["completed"] = list(completed)
            save_progress(progress)
            continue

        entry = {
            "opening": result["name"][0],
            "question": result["question"][0],
            "answer": result["answer"][0],
            "eval": result["eval"][0],
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        completed.add(i)
        progress["completed"] = list(completed)
        progress["failed"] = failed_attempts
        save_progress(progress)

        new_count += 1
        logger.info(f"[{i + 1}/{total}] Saved (eval={eval_score:.1f})")

        time.sleep(2)

    logger.info(
        f"\nDone! New: {new_count}, Skipped: {skip_count}, Total saved: {len(filtered_entries) + new_count}"
    )


if __name__ == "__main__":
    main()
