"""
Updated SageMaker AutoML Pipeline for Dine Brands dataset
Includes DataQuality, ModelQuality, DataBias, ModelBias, Explainability checks,
DriftCheckBaselines and ModelMetrics wired into model registration.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"])
import os
import boto3
import pandas as pd
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# -------------------------
# Helper session functions
# -------------------------
def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket
    )

# -------------------------
# Pipeline definition
# -------------------------
def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    pipeline_name="DineAutoMLPipeline",
    model_package_group_name="DineAutoMLModelPackageGroup",
    output_prefix="dine-auto-ml",
    base_job_prefix="dine_auto_ml",
    sagemaker_project_name="dine_auto_ml_demo"
):
    pipeline_session = get_pipeline_session(region, default_bucket)
    if role is None:
        role = get_execution_role()

    # -------------------------
    # Parameters
    # -------------------------
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    max_automl_runtime = ParameterInteger(name="MaxAutoMLRuntime", default_value=3600)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_registration_metric_threshold = ParameterFloat(name="ModelRegistrationMetricThreshold", default_value=0.8)
    s3_bucket_param = ParameterString(name="S3Bucket", default_value=pipeline_session.default_bucket())
    target_attribute_name = ParameterString(name="TargetAttributeName", default_value="customer churn")

    # Monitoring parameters
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=False)
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value='')

    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value='')

    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=False)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value='')

    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=False)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value='')

    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=False)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=False)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value='')

    # -------------------------
    # Check Job Config
    # -------------------------
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        sagemaker_session=pipeline_session,
    )

    # -------------------------
    # Load dataset and upload to S3
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
    # Preprocessing Step
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
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ],
            code=os.path.join(BASE_DIR, "preprocess.py"),
            arguments=["--input-data", raw_dataset_s3],
        )
    )

    # -------------------------
    # AutoML Training
    # -------------------------
    automl = AutoML(
        role=role,
        target_attribute_name=target_col,
        sagemaker_session=pipeline_session,
        max_candidates=5,
        max_runtime_per_training_job_in_seconds=max_automl_runtime,
        mode="AutoML"
    )
    step_auto_ml_training = AutoMLStep(
        name="AutoMLTrainingStep",
        step_args=automl.fit(inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_col)])
    )

    # -------------------------
    # Model Creation
    # -------------------------
    best_model = step_auto_ml_training.get_best_auto_ml_model(role=role, sagemaker_session=pipeline_session)
    step_create_model = ModelStep(
        name="ModelCreationStep",
        step_args=best_model.create(instance_type=instance_type)
    )

    # -------------------------
    # Batch Transform
    # -------------------------
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type=instance_type,
        output_path=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "transform"]),
        sagemaker_session=pipeline_session,
    )
    step_transform = TransformStep(
        name="BatchTransformStep",
        step_args=transformer.transform(
            data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            input_filter="$[1:]",
            join_source="Input",
            content_type="text/csv",
            split_type="Line"
        )
    )

    # -------------------------
    # Model Evaluation
    # -------------------------
    evaluation_report = PropertyFile(
        name="evaluation",
        output_name="evaluation_metrics",
        path="evaluation_metrics.json"
    )
    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=sklearn_processor.run(
            inputs=[
                ProcessingInput(source=step_transform.properties.TransformOutput.S3OutputPath,
                                destination="/opt/ml/processing/input/predictions"),
                ProcessingInput(source=s3_y_test, destination="/opt/ml/processing/input/true_labels"),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation_metrics",
                    source="/opt/ml/processing/evaluation",
                    destination=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "evaluation"])
                )
            ],
            code=os.path.join(BASE_DIR, "evaluate.py"),
        ),
        property_files=[evaluation_report],
    )

    # -------------------------
    # Conditional Retry if F1 < 0.8
    # -------------------------
    f1_metric = JsonGet(step=step_evaluation, property_file=evaluation_report, json_path="classification_metrics.weighted_f1.value")
    cond_f1_first = ConditionGreaterThanOrEqualTo(f1_metric, 0.8)
    step_cond_first = ConditionStep(
        name="CheckF1ScoreFirst",
        conditions=[cond_f1_first],
        if_steps=[],  # Already passed
        else_steps=[
            AutoMLStep(
                name="AutoMLRetryStep",
                step_args=automl.fit(inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_col)])
            )
        ]
    )

    # -------------------------
    # Model Monitoring Steps (sketched)
    # -------------------------
    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=DataQualityCheckConfig(),
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=ModelQualityCheckConfig(),
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name
    )

    # Bias and Explainability Checks (Clarify)
    data_bias_check_step = ClarifyCheckStep(name="DataBiasCheckStep", clarify_check_config=DataBiasCheckConfig(), check_job_config=check_job_config, skip_check=skip_check_data_bias)
    model_bias_check_step = ClarifyCheckStep(name="ModelBiasCheckStep", clarify_check_config=ModelBiasCheckConfig(), check_job_config=check_job_config, skip_check=skip_check_model_bias)
    model_explainability_check_step = ClarifyCheckStep(name="ModelExplainabilityCheckStep", clarify_check_config=ModelExplainabilityCheckConfig(), check_job_config=check_job_config, skip_check=skip_check_model_explainability)

    # -------------------------
    # DriftCheckBaselines & Model Metrics
    # -------------------------
    drift_check_baselines = DriftCheckBaselines()
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "evaluation", "evaluation_metrics.json"]),
            content_type="application/json"
        )
    )

    # -------------------------
    # Model Registration
    # -------------------------
    register_model_step_args = best_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[instance_type],
        transform_instances=[instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines
    )
    
    step_register_model = ModelStep(
        name="ModelRegistrationStep",
        step_args=register_model_step_args
    )

    # -------------------------
    # Assemble Pipeline
    # -------------------------
    
    pipeline_steps = [
        step_process,
        step_auto_ml_training,
        step_create_model,
        step_transform,
        step_evaluation,
        step_cond_first,
        data_quality_check_step,
        model_quality_check_step,
        data_bias_check_step,
        model_bias_check_step,
        model_explainability_check_step,
        step_register_model,
    ]

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_count, instance_type, max_automl_runtime,
            model_approval_status, model_registration_metric_threshold,
            s3_bucket_param, target_attribute_name,
            skip_check_data_quality, register_new_baseline_data_quality,
            supplied_baseline_statistics_data_quality, supplied_baseline_constraints_data_quality,
            skip_check_model_quality, register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality, supplied_baseline_constraints_model_quality,
            skip_check_data_bias, register_new_baseline_data_bias, supplied_baseline_constraints_data_bias,
            skip_check_model_bias, register_new_baseline_model_bias, supplied_baseline_constraints_model_bias,
            skip_check_model_explainability, register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability
        ],
        steps=pipeline_steps,
        sagemaker_session=pipeline_session
    )

    return pipeline

