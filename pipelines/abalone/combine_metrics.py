import pandas as pd
import os
import glob

if __name__ == "__main__":
    # Input directories are specified by the ProcessingStep
    predictions_dir = "/opt/ml/processing/input/predictions/"
    ground_truth_dir = "/opt/ml/processing/input/ground_truth/"
    output_dir = "/opt/ml/processing/output/combined/"
    
    # The batch transform output might be a single file or partitioned
    prediction_files = glob.glob(f"{predictions_dir}/*.csv*")
    if not prediction_files:
        raise ValueError("No prediction files found. Batch Transform might have failed.")
        
    # Assuming single prediction file and single ground truth file
    predictions_df = pd.read_csv(prediction_files[0], header=None)
    ground_truth_df = pd.read_csv(os.path.join(ground_truth_dir, "y_test.csv"), header=None)
    
    # Combine them side-by-side
    # Predictions will be column 0, ground truth will be column 1
    combined_df = pd.concat([predictions_df, ground_truth_df], axis=1)
    
    # The Model Quality check expects the columns to be named '_c0', '_c1', etc.
    # when header=False, which pandas does by default.
    
    os.makedirs(output_dir, exist_ok=True)
    combined_df.to_csv(os.path.join(output_dir, "metrics.csv"), header=False, index=False)
    
    print("Successfully combined predictions and ground truth.")
