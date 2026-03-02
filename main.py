import json
import os
import time

from datasets import Dataset
from dotenv import load_dotenv
from sdg_hub.core.flow import Flow

# Load seed data and custom flow
dataset = Dataset.from_csv("chess_openings.csv")
flow = Flow.from_yaml("chess_flow/flow.yaml")

# Configure Model and APIs
load_dotenv()  # reads the .env file

flow.set_model_config(
    model="openai/llama-3.1-8b-instant",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
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
