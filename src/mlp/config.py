# src/mlp/config.py
from pathlib import Path

# ==================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent  # Remonte à la racine du projet
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_PROCESSING_OUTPUT = OUTPUT_DIR / "data_processing"
MODEL_OUTPUT = OUTPUT_DIR / "models"
VISU_OUTPUT = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_PROCESSING_OUTPUT.mkdir(exist_ok=True)
MODEL_OUTPUT.mkdir(exist_ok=True)
VISU_OUTPUT.mkdir(exist_ok=True)

# ==================================================================

STAND_CONSTANTS = "constants_stand.csv"
OUTPUT_FILENAMES = [
                "Xy_train.csv",
                "Xy_validation.csv",
]
NETWORK_MLP = "network_structure.csv"
MODEL_PARAMETERS = MODEL_OUTPUT / "model_parameters.npz"

# ==================================================================

FULL = "FULL"
RELEVANT = "RELEVANT"
INDEX_NAME = "ID"
TARGET = "Diagnosis"

# ==================================================================

COLUMNS_NAMES = [
                'ID',
                'Diagnosis',
                # 1. Radius
                'Mean Radius',
                'Radius SE',
                'Worst Radius',
                # 2. Texture
                'Mean Texture',
                'Texture SE',
                'Worst Texture',
                # 3. Perimeter
                'Mean Perimeter',
                'Perimeter SE',
                'Worst Perimeter',
                # 4. Area
                'Mean Area',
                'Area SE', 
                'Worst Area',      
                # 5. Smoothness
                'Mean Smoothness',  
                'Smoothness SE',    
                'Worst Smoothness',
                # 6. Compactness
                'Mean Compactness', 
                'Compactness SE',  
                'Worst Compactness',
                # 7. Concavity
                'Mean Concavity',   
                'Concavity SE',  
                'Worst Concavity',
                # 8. Concave points
                'Mean Concave Points',
                'Concave Points SE', 
                'Worst Concave Points',
                # 9. Symmetry
                'Mean Symmetry',
                'Symmetry SE', 
                'Worst Symmetry',
                # 10. Fractal Dimension
                'Mean Fractal Dimension',
                'Fractal Dimension SE',
                'Worst Fractal Dimension'
]

LIST_THREE_FEAT = [
                'Worst Area',
                'Worst Smoothness', 
                'Mean Texture',
]

LIST_RELEVANT_HISTO = [
                'Mean Radius',
                'Worst Radius',
                'Mean Texture',
                'Texture SE',
                'Worst Texture',
                'Mean Perimeter',
                'Perimeter SE',
                'Worst Perimeter',
                'Area SE', 
                'Worst Area',      
                'Mean Smoothness',  
                'Smoothness SE',    
                'Mean Compactness', 
                'Compactness SE',  
                'Worst Compactness',
                'Concavity SE',  
                'Worst Concavity',
                'Concave Points SE', 
                'Worst Concave Points',
                'Symmetry SE', 
                'Worst Symmetry',
                'Mean Fractal Dimension',
]

TEST_TRAIN_SPLIT_PARAMS = {
                'test_size': 0.25,
                'random_state': 42,
                'shuffle': True,
}

OPTIMIZER = [
                "SGD",
                "ADAM",
]

EARLY_STOPPER = {
                "patience" : 5,
                "delta" : 1e-4,
                "start_from_epoch" : 0,
                "monitor" : "loss",
}