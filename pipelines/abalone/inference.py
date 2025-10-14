# inference.py
import subprocess
import sys
import os

# Force install botocore (and boto3, just in case)
subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "--upgrade", "--quiet"])

def model_fn(model_dir):
    # Your model loading logic here (usually handled by AutoML wrapper)
    # For AutoML, you might just rely on the default handler or add custom logic
    pass

def transform_fn(model, data, input_content_type, accept_content_type):
    # Your transformation logic here
    pass
