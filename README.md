# pb-ai: League of Legends Pick/Ban Outcome Predictor

## Project Vision
This project is part of a larger system to predict the likely outcome of a professional League of Legends pick and ban phase using
machine learning and historical data. The full system will include:
- Automated data ingestion (cronjob)
- Reliable storage for raw and processed data
- Model training and evaluation
- Model output storage and versioning
- A user interface (UI) for interaction and visualization

## Project Structure
```
pb-ai/
  pbai/
    data/
      data_ingestion_service.py
      dataset.py
      processors/
      storage/
    models/
      mlp.py
    training/
      train_oracle_elixir.py
    utils/
      champ_enum.py
      config.py
  scripts/
    ingest_oracle_elixir.py
    run_pipeline.py
    train_oracle_elixir.py
    infer.py
  resources/
    2025_LoL_esports_match_data_from_OraclesElixir.csv
    champions.json
  requirements.txt
  README.md
```

## Setup
1. Clone the repository and navigate to the project root.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the Oracle's Elixir CSV is available in `resources/` (or supply a custom path when running the scripts).

## Workflow Overview
| Stage | Description | Command |
|-------|-------------|---------|
| Data ingestion | Cache the Oracle's Elixir CSV into parquet slices (`data/processed`) for teams and players. | `python scripts/ingest_oracle_elixir.py` |
| Model training | Build the `DraftDataset` and train the `DraftMLP` model, logging metrics along the way. | `python scripts/train_oracle_elixir.py` |
| Inference (placeholder) | Reserved for future model loading and prediction logic. | `python scripts/infer.py` |
| End-to-end pipeline | Run ingestion and training back-to-back with a single command. | `python scripts/run_pipeline.py` |

Each stage can be run independently as you iterate on the pipeline, or chained via the end-to-end script for a complete refresh.

### Stage 1 – Data Ingestion
The ingestion step materializes the CSV export from Oracle's Elixir into versioned parquet files so downstream stages can operate on
fast local reads. Under the hood the script relies on `DataIngestionService` and `FileRepository` to manage caching and filtering.

```bash
python scripts/ingest_oracle_elixir.py \
  --source-path resources/2025_LoL_esports_match_data_from_OraclesElixir.csv \
  --processed-dir data/processed
```

- Use `--force-refresh` to ignore cached parquet files and re-read the CSV.
- Data is written to `data/processed/` with rolling version cleanup to keep disk usage in check.

### Stage 2 – Model Training
Training consumes the cached data, constructs the `DraftDataset`, and optimizes a multilayer perceptron that predicts the next pick or ban.
Loss curves, validation accuracy, and basic sample predictions are printed to aid debugging.

```bash
python scripts/train_oracle_elixir.py \
  --oracle-elixir-path resources/2025_LoL_esports_match_data_from_OraclesElixir.csv \
  --processed-data-dir data/processed \
  --model-path models/draft_mlp_oracle_elixir.pth
```

Adjust `--epochs`, `--batch-size`, or `--learning-rate` to experiment with training dynamics. The best validation checkpoint is saved to the path
supplied via `--model-path` and evaluation metrics are reported against a held-out test split.

### Stage 3 – Inference (coming soon)
The inference CLI stub exists so that future work can plug in trained model weights and champion draft scenarios without reshaping the
project layout. Once implemented it will reuse the cached data schemas and `DraftMLP` definition.

### One-Command Pipeline
To start from raw CSV and finish with a trained checkpoint in one step, use the pipeline script:

```bash
python scripts/run_pipeline.py
```

The script runs ingestion (unless `--skip-ingestion` is provided) and then launches the same training routine as Stage 2. All CLI flags exposed
by the individual stages are available here as well, making it convenient to automate cron-style refreshes or rapid experiments.

## Future Components
- **Data Ingestion:** Automated scripts to fetch and preprocess new data daily.
- **Storage:** Database or cloud storage for raw, processed, and model data.
- **Model Serving:** API endpoints for inference and model management.
- **User Interface:** Web or desktop UI for interacting with predictions and analytics.

---
For more details, see the code in the respective modules.
