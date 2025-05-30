# Air Travel Sentiment Analysis

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

### STEPS:

Clone the repository

```bash
https://github.com/AashishPalSingh/AirTravel_Sentiment_Analysis
```
### Create a venv environment after opening the repository

```bash
python -m venv venv
```

```bash
venv/bin/activate 
source venv/Scripts/Activate
```

### install the requirements
```bash
pip install -r requirements.txt
```

```bash
python init_project.py
```


## Test Logger
```bash
python testlogger.py
```

## setup dagshub and mlflow 

```
export MLFLOW_TRACKING_USERNAME=ashish.student2025
export MLFLOW_TRACKING_PASSWORD=6402a3805769c4b539bb41f9be2830e326045e18
```

## setup dvc 

```
export GOOGLE_APPLICATION_CREDENTIALS=grand-store-457317-n6-9842fd6986c8.json
```

```
dvc init
dvc remote add -d remote gs://dvc-storage-airtravel-sentiment-analysis/storage
dvc add data
dvc push -r remote -v
dvc repro
dvc dag
```


```
streamlit run app.py
```
