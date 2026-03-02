import json
import os
import time

from datasets import Dataset
from dotenv import load_dotenv
from sdg_hub.core.flow import Flow

# Configure Model and APIs
load_dotenv()  # reads the .env file

# MODEL = "openai/llama-3.3-70b-versatile"
MODEL = "openai/llama-3.1-8b-instant"  # really only works with this model!
API_BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")

# Uncomment to use ollama instead!
# MODEL = "openai/llama3.2:1b"
# API_BASE = "http://localhost:11434/v1"
# API_KEY = "ollama"

# Load seed data and custom flow
dataset = Dataset.from_csv("chess_openings.csv")
flow = Flow.from_yaml("chess_flow/flow.yaml")

flow.set_model_config(
    model=MODEL,
    api_base=API_BASE,
    api_key=API_KEY,
)

print("\nDry run...")
dry = flow.dry_run(dataset, sample_size=1)
print(f"Passed in {dry['execution_time_seconds']:.2f}s")

output_file = "chess_openings_qa.jsonl"
with open(output_file, "w") as f:
    for i in range(len(dataset)):
        result = flow.generate(dataset.select([i]))

        f.write(
            json.dumps(
                {
                    "opening": result["name"][0],
                    "question": result["question"][0],
                    "answer": result["answer"][0],
                    "eval": result["eval"][0],
                }
            )
            + "\n"
        )
        f.flush()
        time.sleep(2)  # Sleep to reduce request count

print(f"\nSaved Q&A pairs to '{output_file}'")
