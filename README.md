# pb-ai: League of Legends Pick/Ban Outcome Predictor

## Project Vision
This project is part of a larger system to predict the likely outcome of a professional League of Legends pick and ban phase, using machine learning and historical data. The full system will include:
- Automated data ingestion (cronjob)
- Reliable storage for raw and processed data
- Model training and evaluation
- Model output storage and versioning
- A user interface (UI) for interaction and visualization

## Project Structure
```
pb-ai/
  pbai/
    __init__.py
    data/
      __init__.py
      dataset.py
      loader.py
    models/
      __init__.py
      mlp.py
    training/
      __init__.py
      train.py
    utils/
      __init__.py
      champ_enum.py
      config.py
  scripts/
    train.py
    infer.py
  resources/
    champions.json
    fp-data.csv
    fp-data2.csv
  requirements.txt
  README.md
```

## Setup
1. Clone the repository and navigate to the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure data files are in the `resources/` directory.

## Training the Model
Run the training script:
```bash
python scripts/train.py
```

## Future Components
- **Data Ingestion:** Automated scripts to fetch and preprocess new data daily.
- **Storage:** Database or cloud storage for raw, processed, and model data.
- **Model Serving:** API endpoints for inference and model management.
- **User Interface:** Web or desktop UI for interacting with predictions and analytics.

---
For more details, see the code in the respective modules.
