# BridgeBART: Real-Time Multi-Modal Video Captioning and Transition Modeling

This repository contains our capstone project exploring real-time video captioning with
BLIP-2 for frame captions, a teacher-student pipeline for semantic transitions (GPT to BART),
and multi-model summarization for concise video descriptions. See `ML_Final_Project.pdf`
for the full methodology and results.

## Highlights
- Frame sampling and captioning with BLIP-2.
- Transition modeling using a fine-tuned BART model ("bridgeBART").
- Optional GPT-based summaries for higher-level narrative synthesis.
- Evaluation utilities and exploration scripts in `Workbench/`.
- Legacy experiments preserved in `Deprecated/`.

## Repository Layout
- `BLIP2Vid2.py`: Main end-to-end pipeline.
- `bridgeBART_builder.py`: Training pipeline for the bridgeBART model.
- `metrics.py`: Evaluation helpers and metrics experiments.
- `Training Data/`: Human-labeled transition data + comparison charts.
- `Workbench/`: Exploration tools, scripts, and derived datasets.
- `Deprecated/`: Older experiments and prototypes.
- `ML_Final_Project.pdf`: Final report.
- `Benchmark Videos/03.mp4`: Sample input video (other benchmark videos not included).
- `examples/`: Small sample outputs for quick review.

## Quick Start
1) Create an environment (Python 3.10+ recommended).
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) (Optional) Set your OpenAI key for GPT-based summaries (environment variable or `.env`):
```bash
setx OPENAI_API_KEY "your_key_here"
```
4) Run the pipeline:
```bash
python BLIP2Vid2.py
```

## Runtime Notes
- GPU is strongly recommended for real-time or near real-time throughput; CPU runs will be much slower.
- `requirements.txt` is pinned to CUDA-enabled `torch`/`torchvision` builds; use CPU-only wheels if you don't have CUDA.

## Training bridgeBART
The transition model trains from `Training Data/transition_sheet.csv` and saves into
`Models/bridgeBART/` (ignored by git):
```bash
python -c "from bridgeBART_builder import run_bridgebart_training_pipeline; run_bridgebart_training_pipeline()"
```

## Data Notes
- The training CSV contains caption pairs and human transition labels derived from the
  project dataset. See the report for dataset citations.
- Only a single sample video is included. Add your own videos under `Benchmark Videos/`
  to run additional tests.

## Results
The final report contains the full analysis. The repo includes the model comparison charts:
- [Accuracy](Training%20Data/Model_Accuracy_Comparison.png)
- [Cosine similarity](Training%20Data/Model_Cosine_Similarity_Comparison.png)
- [Fluency](Training%20Data/Model%20Fluency%20Comparison.png)

## Example Output
See `examples/generated_transitions_sample.csv` for a small sample of generated
transitions from the workbench pipeline.

## License
All rights reserved. See `LICENSE`.

## Authors
Kelsey Knowlson, Matthew Rackley
