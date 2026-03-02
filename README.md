# Chess Opening Q&A Generator

A synthetic data generation pipeline for chess opening strategy, built using [Red Hat AI Innovation Team's](https://ai-innovation.team/) `sdg_hub` library.

This project generates high-quality question and answer pairs about chess openings from seed documents. The generated data can be used to fine-tune a small language model to better understand chess opening strategy.

## What It Does

Large language models generally struggle with chess — particularly with understanding *why* moves are made rather than just listing them. This project addresses that by generating targeted Q&A pairs focused on the strategic reasoning behind chess openings.

The pipeline:
1. Reads chess opening descriptions from `chess_openings.csv`
2. Sends each set of opening data through a custom `sdg_hub` flow
3. A teacher LLM generates a question and answer for each opening
4. The same LLM also rates the Q&A pair from 1-10
4. Results are saved to `chess_openings_qa.jsonl`
