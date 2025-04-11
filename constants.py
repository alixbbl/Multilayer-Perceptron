FULL = "FULL"
RELEVANT = "RELEVANT"

INDEX_NAME = "ID"
COLUMNS_NAMES = [
                'ID',       # Id
                'Diagnosis',        # Diagnostic (M = maligne, B = bénigne)
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

TARGET = "Diagnosis"
TEST_TRAIN_SPLIT_PARAMS = {
                'test_size' : 0.25,
                'random_state' : 42,
                'shuffle' : True,
}

OUTPUT_FILENAMES = [
                "Xy_train.csv",
                "Xy_test.csv",
]
