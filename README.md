
# OpenTraj: An Open-Source Platform for Generalized Trajectory Mining Using Large Language Models

The OpenTraj Platform is a comprehensive system for trajectory analysis and prediction, providing three key functionalities:
- Destination Prediction (DP): Predicts the final destination of moving objects
- Travel Time Estimation (TTE): Estimates remaining travel time to destination
- Similar Trajectory Search (STS): Finds similar historical trajectories

# Project structure
```
project/
├── app.py                # Flask web application
├── data.py               # Data preprocessing and loading
├── main.py               # Model training entry point
├── model/                # Model implementations
│   ├── let.py            # LET trajectory embedding model
│   └── layers.py         # Neural network components
├── downstream/           # Task-specific modules
│   ├── task.py           # Downstream task trainers
│   └── predictor.py      # Prediction heads
├── loss/                 # Loss functions
├── dataloader/           # Data loading utilities
├── pretrain/             # Pre-training modules
├── utils/                # Utility functions
├── sample/               # Dataset for training models
├── model_save/           # Trained model files
├── templates/            # HTML templates
└── use_case/             # Sample datasets
```

## Getting Started

### Prerequisites
- Python 3.12
- PyTorch 2.3.0
- Flask
- AMap JavaScript API

### Data Preparation
We provide an example dataset(small_chengdu.h5) for debugging purposes with the same format as our full dataset.    The dataset is stored in HDF5 format with the following structure:
```
/trip           # Map-matched GPS sequences (longitude, latitude, timestamp, segment ID, road type, lanes, etc.)
/trip_info      # Trajectory metadata (length, start/end time)
/poi            # Point-of-interest (POI) data
/road_info      # Road network data (extracted from OpenStreetMap)
```
### Custom Dataset
To use custom datasets, preprocess your data as follows:
1.    **Road Network**: Extract coverage from OpenStreetMap(https://www.openstreetmap.org/)
2.    **POI Data**: Acquire via Amap APIs ([Amap APIs](https://lbs.amap.com/))
3.    **Trip Metadata**: Automatically generated from trajectory files


Preprocess with:
```bash
python data.py -n small_chengdu -t trip,odpois-3,destination,tte,hopqrytgtpois-1000-3,hopqrytgt-1000,hopnegindex-1000-5000 -i 0,1,2
```

### Model Training
Train models using configuration files:
```bash
python main.py --config small_chengdu --cuda 0
```

### Running the Web Application
Start the Flask server:
```bash
python app.py
```
