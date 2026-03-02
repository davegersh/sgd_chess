# Chess Opening Q&A Generator

A synthetic data generation pipeline for chess opening strategy, built using [Red Hat AI Innovation Team's](https://ai-innovation.team/) `sdg_hub` library.

This project generates high-quality question and answer pairs about chess openings from seed documents. The generated data can be used to fine-tune a small language model to better understand chess opening strategy.

## What It Does

Large language models generally struggle with chess — particularly with understanding *why* moves are made rather than just listing them. This project addresses that by generating targeted Q&A pairs focused on the strategic reasoning behind chess openings.

The pipeline:
1. Reads chess opening descriptions from `chess_openings.csv`
2. Sends each set of opening data through a custom `sdg_hub` flow
3. A teacher LLM generates a question and answer for each opening
4. The same LLM also rates the Q&A pair from 1-10.
4. Results are saved to `chess_openings_qa.jsonl`

## Running

### Dependencies
- Make sure you have installed [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Local ollama instance or cloud inference (like Groq) API key.

### Setting up
To run the pipeline, you'll need to setup the LLM you are going to use. 
In `main.py`, after the flow is created, feel free to modify the `set_model_config` arguments to fit your needs.
By default, it's using a the Groq API.

### Run the pipeline
```
uv run python main.py
```

Q&A pairs are saved to `chess_openings_qa.jsonl` as they are generated. If the script crashes midway, already-saved pairs are not lost.


### Project files

```
main.py                           # Main script — run this
chess_flow/flow.yaml              # Custom sdg_hub flow definition
chess_flow/prompts/qa_gen.yaml    # Prompt template for Q&A generation
chess_flow/prompts/eval.yaml      # Prompt template for eval generation
chess_openings.csv                # Seed data — 55 chess openings
chess_openings_qa.jsonl           # Output — generated Q&A pairs (created on run)
```

## The Development Process
This project was fully focused on learning the ins and outs of how Red Hat's `sdg_hub` works specifically on a problem that I've seen commonly apear in LLMs: chess knowledge!

### AI Usage
This was a project that involved extensive conversation with Claude (Sonnet 4.6). It was an interesting experience that was very much a "move fast, break things" approach. With heavy enphasis on the "break things" part. When Claude worked, it worked really well and it was fantastic for getting a simple script working and having a foundation to work with. It was only until I had asked for more details about `sdg_hub` that I think it struggled a bit. For example, at one point it mentioned a "known bug" existed in flow creation with `sdg_hub`. When prompted for it's source, it mentioned that it "made it up." I believe that the AI usage was at it's best when I took the reigns and asked accessory or supportive questions rather than generating too much all at once. Although this may be a sign that I'm due to work on improving my prompting skills.

Claude also helped generate the set of 55 openings and their descriptions.

### What Worked
The built-in `sdg_hub` flows worked reliably out of the box. The `FlowRegistry` pattern for discovering and loading pre-built flows is clean and well-designed. Pointing it at any OpenAI-compatible endpoint is straightforward.

The custom YAML flow ended up working once I understood the correct block pattern from reading the actual installed flow YAML files. The key insight was that `LLMChatBlock` requires a `PromptBuilderBlock` before it to format the prompt as a structured message list — it cannot accept a plain string column directly. The pattern is always: `PromptBuilderBlock` → `LLMChatBlock` → `LLMResponseExtractorBlock` → `TagParserBlock`.

Moving seed data into a CSV made the project much cleaner. Adding new openings requires no code changes — just a new row in the CSV.

### What didn't work
Running Ollama locally was slow without GPU acceleration. I had initially thought my laptop was more than capable even on just the CPU-side but I found it slow, even with a simpler custom flow. For local development, using a free cloud API like Groq was significantly more practical as long as I stayed under the free-tier limits.

The built-in flow token usage was too high for Groq's free tier rate limits when using long seed documents. The solution was to keep seed documents short and concise — 3-4 sentences per opening rather than full paragraphs.

Smaller models with ~1B parameters struggled to follow the directions in the flow prompts. Highly reccommended to use larger models especially as they'll generate better Q&A pairs too.

## Other Tools and Tradeoffs
### Groq API
Free cloud inference for the teacher model.
- **Pros:** Genuinely free tier, very fast inference, runs large models like Llama 3.3 70B, OpenAI-compatible API
- **Cons:** Rate limits on the free tier require throttling requests, not suitable for large-scale production generation without a paid plan

### Ollama
Local LLM inference.
- **Pros:** Completely free, no API key, no rate limits, data never leaves your machine
- **Cons:** Requires decent hardware to run useful models at a reasonable speed, my laptop was not considered decent enough

## Next Steps
Add fine-tuning with `training_hub`. The natural next step is to feed the generated Q&A pairs into Red Hat's `training_hub` to fine-tune a small model and demonstrate a measurable improvement in chess opening reasoning with a before/after comparison.

Add the ability to eliminate low-scoring Q&A pairs and improve the eval prompt. Currently, it generates a single number, however, other evaluation methods could work better. Eliminating low-scoring Q&A pairs should be as simple as adding a filter block to the flow.

Support multiple Q&A pairs per opening. Currently the flow generates one Q&A pair per opening document. Running the flow multiple times per document with different temperature settings or prompt variations would produce a richer, more diverse dataset with better eval scores.

Expand beyond openings. For the sake of simplicity, this project focuses on openings. With more time I could generate more detailed descriptions of each opening or move beyond and generate middle/endgame tactics.
