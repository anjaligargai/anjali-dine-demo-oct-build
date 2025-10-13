"""
Updated SageMaker AutoML Pipeline for Dine Brands dataset
Includes AutoML training, retries, batch transform, evaluation, 
conditions, model registration, and model monitoring.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"])

import os
import boto3
import pandas as pd
from io import StringIO

from sklearn.model_selection import train_test_split

import sagemaker
from sagemaker import AutoML, AutoMLInput, get_execution_role
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.quality_check_step import QualityCheckStep
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.drift_check_baselines import DriftCheckBaselines
from sagemaker.model_monitor import DatasetFormat
from sagemaker.clarify import DataConfig, BiasConfig, ModelConfig, ModelPredictedLabelConfig, SHAPConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelExplainabilityCheckConfig,
    ModelQualityCheckConfig,
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# -------------------------
# Helper functions
# -------------------------
def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

# -------------------------
# Main pipeline function
# -------------------------
def get_pipeline(
    region,
    role,
    default_bucket,
    pipeline_name="DineAutoMLTrainingPipeline",
    model_package_group_name="AutoMLModelPackageGroup",
    output_prefix="dine-auto-ml-training",
    base_job_prefix="mlops_dine",
    sagemaker_project_name="mlops_dine_demo"
):

    pipeline_session = get_pipeline_session(region, default_bucket)
    if role is None:
        role = get_execution_role()

    # -------------------------
    # Pipeline parameters
    # -------------------------
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    max_automl_runtime = ParameterInteger(name="MaxAutoMLRuntime", default_value=3600)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_registration_metric_threshold = ParameterFloat(name="ModelRegistrationMetricThreshold", default_value=0.8)
    s3_bucket_param = ParameterString(name="S3Bucket", default_value=pipeline_session.default_bucket())
    target_attribute_name = ParameterString(name="TargetAttributeName", default_value="customer churn")

    # -------------------------
    # Load dataset from S3
    # -------------------------
    raw_dataset_s3 = "s3://aishwarya-mlops-demo/dine_customer_churn/dine_data/dataset1_20k.csv"
    bucket = raw_dataset_s3.split("/")[2]
    key = "/".join(raw_dataset_s3.split("/")[3:])
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    feature_names = [
        "date","store_id","store_name","city","state","store_type",
        "item_id","item_name","category","price","quantity_sold",
        "revenue","food_cost","profit","day_of_week","month",
        "quarter","is_weekend","is_holiday","temperature","is_promotion",
        "stock_out","prep_time","calories","is_vegetarian",
    ]
    target_col = target_attribute_name.default_value
    df.columns = feature_names + [target_col]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("train_val.csv", index=False)
    test_df[feature_names].to_csv("x_test.csv", index=False, header=False)
    test_df[[target_col]].to_csv("y_test.csv", index=False, header=False)

    prepared_prefix = f"{output_prefix}/prepared"
    train_val_s3_key = f"{prepared_prefix}/train_val.csv"
    x_test_s3_key = f"{prepared_prefix}/x_test/x_test.csv"
    y_test_s3_key = f"{prepared_prefix}/y_test/y_test.csv"

    s3.upload_file("train_val.csv", bucket, train_val_s3_key)
    s3.upload_file("x_test.csv", bucket, x_test_s3_key)
    s3.upload_file("y_test.csv", bucket, y_test_s3_key)

    s3_train_val = f"s3://{bucket}/{train_val_s3_key}"
    s3_x_test_prefix = f"s3://{bucket}/{prepared_prefix}/x_test/"
    s3_y_test = f"s3://{bucket}/{y_test_s3_key}"

    # -------------------------
    # Data quality / bias / monitoring parameters
    # -------------------------
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=False)
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value='')

    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=False)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value='')

    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value='')

    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=False)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value='')

    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=False)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=False)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value='')

    # -------------------------
    # Preprocessing step
    # -------------------------
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="PreprocessDineBrandsData",
        step_args=sklearn_processor.run(
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
                ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
            ],
            code=os.path.join(BASE_DIR, "preprocess.py"),
            arguments=["--input-data", raw_dataset_s3],
        )
    )

    # -------------------------
    # AutoML training step
    # -------------------------
    automl = AutoML(
        role=role,
        target_attribute_name=target_col,
        sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=max_automl_runtime,
        mode="ENSEMBLING"
    )
    step_auto_ml_training = AutoMLStep(
        name="AutoMLTrainingStep",
        step_args=automl.fit(inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_col)])
    )

    # Create model
    best_model = step_auto_ml_training.get_best_auto_ml_model(role, sagemaker_session=pipeline_session)
    step_create_model = ModelStep(name="ModelCreationStep", step_args=best_model.create(instance_type=instance_type))

    # Batch transform
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "transform"]),
        sagemaker_session=pipeline_session,
    )
    step_batch_transform = TransformStep(
        name="BatchTransformStep",
        step_args=transformer.transform(data=s3_x_test_prefix, content_type="text/csv")
    )

    # Evaluation
    evaluation_report = PropertyFile(name="evaluation", output_name="evaluation_metrics", path="evaluation_metrics.json")
    sklearn_processor_eval = SKLearnProcessor(
        role=role,
        framework_version="1.0-1",
        instance_count=instance_count,
        instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
    )
    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=sklearn_processor_eval.run(
            inputs=[
                ProcessingInput(source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                                destination="/opt/ml/processing/input/predictions"),
                ProcessingInput(source=s3_y_test, destination="/opt/ml/processing/input/true_labels"),
            ],
            outputs=[ProcessingOutput(output_name="evaluation_metrics",
                                      source="/opt/ml/processing/evaluation",
                                      destination=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "evaluation"]))],
            code="pipelines/abalone/evaluate.py",
        ),
        property_files=[evaluation_report],
    )

    # -------------------------
    # Condition: F1 >= threshold
    # -------------------------
    f1_metric = JsonGet(
        step=step_evaluation,
        property_file=evaluation_report,
        json_path="classification_metrics.weighted_f1.value"
    )
    cond_f1_first = Condition
