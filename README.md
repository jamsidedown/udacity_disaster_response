# Disaster Response Pipeline Project

## Introduction

This repository holds my work towards the Udacity disaster response pipeline project.

The `src` folder contains three Python modules for data processing, classifier training, and a Flask webapp to display findings and interactions.

## Setup

This project has been developed using Python 3.8.5, to install dependencies create a virtual environment using `venv` or `pyenv` and run
```sh
pip install -r requirements.txt
```

## Usage

The processed data and trained model have not been included in this repository to keep the repository size to a minimum.

To run the ETL pipeline that cleans data and stores it in an Sqlite database at `data/disaster_response.db`
```sh
python -m src.process_data

# or with custom paths
python -m src.process_data data/disaster_messages.csv data/disaster_categories.csv disaster_response.db
```

To run the ML pipeline to train a classifier and save it using Pickle at `models/classifier.pkl`
```sh
python -m src.train_classifier

# or with custom paths
python -m src.train_classifier data/disaster_response.db models/classifier.pkl
```

To run the ML pipeline with GridSearchCV to find the optimal parameters and evaluate the model (this takes a while to run)
```sh
export GRIDSEARCH=true

python -m src.train_classifier
```

To run the webapp using the trained classifier
```sh
python -m src.run
```
and open a web browser at http://localhost:3001

