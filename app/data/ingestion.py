"""
Data ingestion and validation modules for banking data processing.
Supports multiple formats (CSV, JSON, Parquet) with comprehensive validation.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import pyarrow.parquet as pq
from enum import Enum

from ..core.interfaces import DataProcessor
from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class DataFormat(Enum):
    """Supported data formats."""

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Data validation issue."""

    column: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    count: int
    percentage: float
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class DataProfile:
    """Data profiling results."""

    total_rows: int
    total_columns: int
    missing_data_percentage: float
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    column_profiles: Dict[str, Dict[str, Any]]
    data_quality_score: float


@dataclass
class IngestionResult:
    """Result of data ingestion process."""

    success: bool
    data: Optional[pd.DataFrame]
    validation_issues: List[ValidationIssue]
    data_profile: Optional[DataProfile]
    processing_time_seconds: float
    rows_processed: int
    message: str


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def read_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Read data from source."""
        pass

    @abstractmethod
    def validate_source(self, source_path: str) -> bool:
        """Validate if source is accessible and readable."""
        pass


class CSVDataSource(DataSource):
    """CSV data source implementation."""

    def read_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV data."""
        try:
            # Default CSV reading parameters
            csv_params = {
                "encoding": "utf-8",
                "low_memory": False,
                "na_values": ["", "NULL", "null", "NA", "na", "N/A", "n/a"],
                **kwargs,
            }

            data = pd.read_csv(source_path, **csv_params)
            logger.info(
                f"Successfully read CSV file: {source_path}, shape: {data.shape}"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to read CSV file {source_path}: {e}")
            raise

    def validate_source(self, source_path: str) -> bool:
        """Validate CSV source."""
        path = Path(source_path)
        return path.exists() and path.suffix.lower() == ".csv"


class JSONDataSource(DataSource):
    """JSON data source implementation."""

    def read_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Read JSON data."""
        try:
            # Support both JSON lines and regular JSON
            if kwargs.get("lines", False):
                data = pd.read_json(source_path, lines=True, **kwargs)
            else:
                data = pd.read_json(source_path, **kwargs)

            logger.info(
                f"Successfully read JSON file: {source_path}, shape: {data.shape}"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to read JSON file {source_path}: {e}")
            raise

    def validate_source(self, source_path: str) -> bool:
        """Validate JSON source."""
        path = Path(source_path)
        return path.exists() and path.suffix.lower() in [".json", ".jsonl"]


class ParquetDataSource(DataSource):
    """Parquet data source implementation."""

    def read_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet data."""
        try:
            data = pd.read_parquet(source_path, **kwargs)
            logger.info(
                f"Successfully read Parquet file: {source_path}, shape: {data.shape}"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to read Parquet file {source_path}: {e}")
            raise

    def validate_source(self, source_path: str) -> bool:
        """Validate Parquet source."""
        path = Path(source_path)
        return path.exists() and path.suffix.lower() == ".parquet"


class BankingDataValidator:
    """Validator for banking/credit risk data."""

    def __init__(self):
        self.required_columns = [
            "customer_id",
            "age",
            "annual_income_inr",
            "loan_amount_inr",
            "credit_score",
            "debt_to_income_ratio",
            "default",
        ]

        self.categorical_columns = [
            "gender",
            "marital_status",
            "employment_type",
            "loan_purpose",
        ]

        self.numeric_columns = [
            "age",
            "annual_income_inr",
            "loan_amount_inr",
            "loan_term_months",
            "credit_score",
            "num_open_credit_accounts",
            "num_delinquent_accounts",
            "debt_to_income_ratio",
            "past_default",
            "default",
        ]

        # Define validation rules
        self.validation_rules = {
            "age": {"min": 18, "max": 100},
            "annual_income_inr": {"min": 0, "max": 100000000},
            "loan_amount_inr": {"min": 0, "max": 50000000},
            "credit_score": {"min": 300, "max": 850},
            "debt_to_income_ratio": {"min": 0, "max": 10},
            "loan_term_months": {"min": 1, "max": 480},
            "num_open_credit_accounts": {"min": 0, "max": 50},
            "num_delinquent_accounts": {"min": 0, "max": 20},
            "past_default": {"values": [0, 1]},
            "default": {"values": [0, 1]},
        }

        self.categorical_rules = {
            "gender": ["Male", "Female", "Other"],
            "marital_status": ["Single", "Married", "Divorced", "Widowed"],
            "employment_type": [
                "Salaried",
                "Self-Employed",
                "Government",
                "Unemployed",
            ],
            "loan_purpose": [
                "Personal Loan",
                "Home Loan",
                "Education Loan",
                "Credit Card",
                "Business Loan",
            ],
        }

    def validate_data(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Perform comprehensive data validation."""
        issues = []

        # Check required columns
        issues.extend(self._validate_required_columns(data))

        # Check data types
        issues.extend(self._validate_data_types(data))

        # Check missing values
        issues.extend(self._validate_missing_values(data))

        # Check duplicates
        issues.extend(self._validate_duplicates(data))

        # Check value ranges
        issues.extend(self._validate_value_ranges(data))

        # Check categorical values
        issues.extend(self._validate_categorical_values(data))

        # Check business logic
        issues.extend(self._validate_business_logic(data))

        return issues

    def _validate_required_columns(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate required columns are present."""
        issues = []
        missing_columns = set(self.required_columns) - set(data.columns)

        for column in missing_columns:
            issues.append(
                ValidationIssue(
                    column=column,
                    issue_type="missing_column",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Required column '{column}' is missing",
                    count=1,
                    percentage=100.0,
                )
            )

        return issues

    def _validate_data_types(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data types."""
        issues = []

        for column in self.numeric_columns:
            if column in data.columns:
                non_numeric = data[column].apply(
                    lambda x: not pd.api.types.is_numeric_dtype(type(x)) and pd.notna(x)
                )
                if non_numeric.any():
                    count = non_numeric.sum()
                    percentage = (count / len(data)) * 100

                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="invalid_data_type",
                            severity=ValidationSeverity.ERROR,
                            message=f"Column '{column}' contains non-numeric values",
                            count=count,
                            percentage=percentage,
                            sample_values=data[column][non_numeric].head(5).tolist(),
                        )
                    )

        return issues

    def _validate_missing_values(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate missing values."""
        issues = []

        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                percentage = (missing_count / len(data)) * 100
                severity = (
                    ValidationSeverity.CRITICAL
                    if column in self.required_columns
                    else ValidationSeverity.WARNING
                )

                issues.append(
                    ValidationIssue(
                        column=column,
                        issue_type="missing_values",
                        severity=severity,
                        message=f"Column '{column}' has missing values",
                        count=missing_count,
                        percentage=percentage,
                    )
                )

        return issues

    def _validate_duplicates(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate duplicate records."""
        issues = []

        # Check for duplicate customer IDs
        if "customer_id" in data.columns:
            duplicate_ids = data["customer_id"].duplicated().sum()
            if duplicate_ids > 0:
                percentage = (duplicate_ids / len(data)) * 100
                issues.append(
                    ValidationIssue(
                        column="customer_id",
                        issue_type="duplicate_values",
                        severity=ValidationSeverity.ERROR,
                        message="Duplicate customer IDs found",
                        count=duplicate_ids,
                        percentage=percentage,
                    )
                )

        # Check for completely duplicate rows
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            percentage = (duplicate_rows / len(data)) * 100
            issues.append(
                ValidationIssue(
                    column="all_columns",
                    issue_type="duplicate_rows",
                    severity=ValidationSeverity.WARNING,
                    message="Duplicate rows found",
                    count=duplicate_rows,
                    percentage=percentage,
                )
            )

        return issues

    def _validate_value_ranges(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate numeric value ranges."""
        issues = []

        for column, rules in self.validation_rules.items():
            if column not in data.columns:
                continue

            if "min" in rules and "max" in rules:
                out_of_range = (data[column] < rules["min"]) | (
                    data[column] > rules["max"]
                )
                out_of_range = out_of_range & data[column].notna()

                if out_of_range.any():
                    count = out_of_range.sum()
                    percentage = (count / len(data)) * 100

                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="out_of_range",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{column}' has values outside range [{rules['min']}, {rules['max']}]",
                            count=count,
                            percentage=percentage,
                            sample_values=data[column][out_of_range].head(5).tolist(),
                        )
                    )

            elif "values" in rules:
                invalid_values = (
                    ~data[column].isin(rules["values"]) & data[column].notna()
                )

                if invalid_values.any():
                    count = invalid_values.sum()
                    percentage = (count / len(data)) * 100

                    issues.append(
                        ValidationIssue(
                            column=column,
                            issue_type="invalid_values",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{column}' has invalid values. Expected: {rules['values']}",
                            count=count,
                            percentage=percentage,
                            sample_values=data[column][invalid_values]
                            .unique()
                            .tolist()[:5],
                        )
                    )

        return issues

    def _validate_categorical_values(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate categorical values."""
        issues = []

        for column, valid_values in self.categorical_rules.items():
            if column not in data.columns:
                continue

            invalid_values = ~data[column].isin(valid_values) & data[column].notna()

            if invalid_values.any():
                count = invalid_values.sum()
                percentage = (count / len(data)) * 100

                issues.append(
                    ValidationIssue(
                        column=column,
                        issue_type="invalid_categorical",
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{column}' has invalid categorical values. Expected: {valid_values}",
                        count=count,
                        percentage=percentage,
                        sample_values=data[column][invalid_values]
                        .unique()
                        .tolist()[:5],
                    )
                )

        return issues

    def _validate_business_logic(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate business logic rules."""
        issues = []

        # Check if loan amount is reasonable compared to income
        if all(col in data.columns for col in ["loan_amount_inr", "annual_income_inr"]):
            loan_to_income_ratio = data["loan_amount_inr"] / data["annual_income_inr"]
            unreasonable_ratio = (
                loan_to_income_ratio > 10
            )  # More than 10x annual income

            if unreasonable_ratio.any():
                count = unreasonable_ratio.sum()
                percentage = (count / len(data)) * 100

                issues.append(
                    ValidationIssue(
                        column="loan_amount_inr",
                        issue_type="business_logic_violation",
                        severity=ValidationSeverity.WARNING,
                        message="Loan amount is more than 10x annual income",
                        count=count,
                        percentage=percentage,
                    )
                )

        # Check if debt-to-income ratio is consistent
        if all(
            col in data.columns
            for col in ["debt_to_income_ratio", "loan_amount_inr", "annual_income_inr"]
        ):
            calculated_ratio = data["loan_amount_inr"] / data["annual_income_inr"]
            ratio_difference = abs(data["debt_to_income_ratio"] - calculated_ratio)
            inconsistent_ratio = ratio_difference > 0.5  # Allow some tolerance

            if inconsistent_ratio.any():
                count = inconsistent_ratio.sum()
                percentage = (count / len(data)) * 100

                issues.append(
                    ValidationIssue(
                        column="debt_to_income_ratio",
                        issue_type="inconsistent_calculation",
                        severity=ValidationSeverity.INFO,
                        message="Debt-to-income ratio may be inconsistent with loan amount and income",
                        count=count,
                        percentage=percentage,
                    )
                )

        return issues


class DataProfiler:
    """Automated data profiling and exploratory data analysis."""

    def profile_data(self, data: pd.DataFrame) -> DataProfile:
        """Generate comprehensive data profile."""
        # Basic statistics
        total_rows = len(data)
        total_columns = len(data.columns)
        missing_data_percentage = (
            data.isnull().sum().sum() / (total_rows * total_columns)
        ) * 100
        duplicate_rows = data.duplicated().sum()

        # Column type classification
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_columns = data.select_dtypes(include=["datetime64"]).columns.tolist()

        # Column-level profiling
        column_profiles = {}
        for column in data.columns:
            column_profiles[column] = self._profile_column(data[column])

        # Calculate data quality score
        data_quality_score = self._calculate_quality_score(data, column_profiles)

        return DataProfile(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_data_percentage=missing_data_percentage,
            duplicate_rows=duplicate_rows,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            column_profiles=column_profiles,
            data_quality_score=data_quality_score,
        )

    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile individual column."""
        profile = {
            "dtype": str(series.dtype),
            "missing_count": series.isnull().sum(),
            "missing_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "unique_percentage": (series.nunique() / len(series)) * 100,
        }

        if pd.api.types.is_numeric_dtype(series):
            profile.update(
                {
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                    "skewness": series.skew(),
                    "kurtosis": series.kurtosis(),
                }
            )
        else:
            # Categorical/text columns
            value_counts = series.value_counts().head(10)
            profile.update(
                {
                    "top_values": value_counts.to_dict(),
                    "most_frequent": (
                        series.mode().iloc[0] if not series.mode().empty else None
                    ),
                }
            )

        return profile

    def _calculate_quality_score(
        self, data: pd.DataFrame, column_profiles: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate overall data quality score (0-100)."""
        scores = []

        # Completeness score (based on missing data)
        completeness = (
            100 - (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        )
        scores.append(completeness)

        # Uniqueness score (based on duplicates)
        uniqueness = 100 - (data.duplicated().sum() / len(data)) * 100
        scores.append(uniqueness)

        # Consistency score (based on data types)
        consistency = 100  # Start with perfect score
        for column in data.columns:
            if column in column_profiles:
                # Penalize high missing percentages
                missing_pct = column_profiles[column]["missing_percentage"]
                if missing_pct > 50:
                    consistency -= 20
                elif missing_pct > 20:
                    consistency -= 10
                elif missing_pct > 5:
                    consistency -= 5

        consistency = max(0, consistency)
        scores.append(consistency)

        return np.mean(scores)


class DataIngestionProcessor(DataProcessor):
    """Main data ingestion processor with validation and profiling."""

    def __init__(self):
        self.config = get_config()
        self.data_sources = {
            DataFormat.CSV: CSVDataSource(),
            DataFormat.JSON: JSONDataSource(),
            DataFormat.PARQUET: ParquetDataSource(),
        }
        self.validator = BankingDataValidator()
        self.profiler = DataProfiler()

    def process(
        self,
        source_path: str,
        data_format: Optional[DataFormat] = None,
        validate: bool = True,
        profile: bool = True,
        **kwargs,
    ) -> IngestionResult:
        """Process data ingestion with validation and profiling."""
        start_time = datetime.now()

        try:
            # Auto-detect format if not provided
            if data_format is None:
                data_format = self._detect_format(source_path)

            # Validate source
            data_source = self.data_sources.get(data_format)
            if not data_source:
                return IngestionResult(
                    success=False,
                    data=None,
                    validation_issues=[],
                    data_profile=None,
                    processing_time_seconds=0,
                    rows_processed=0,
                    message=f"Unsupported data format: {data_format}",
                )

            if not data_source.validate_source(source_path):
                return IngestionResult(
                    success=False,
                    data=None,
                    validation_issues=[],
                    data_profile=None,
                    processing_time_seconds=0,
                    rows_processed=0,
                    message=f"Invalid or inaccessible source: {source_path}",
                )

            # Read data
            data = data_source.read_data(source_path, **kwargs)

            # Validate data
            validation_issues = []
            if validate:
                validation_issues = self.validator.validate_data(data)

            # Profile data
            data_profile = None
            if profile:
                data_profile = self.profiler.profile_data(data)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log ingestion
            audit_logger.log_data_access(
                user_id="system",
                resource=source_path,
                action="data_ingestion",
                success=True,
                details={
                    "format": data_format.value,
                    "rows": len(data),
                    "columns": len(data.columns),
                    "validation_issues": len(validation_issues),
                    "processing_time_seconds": processing_time,
                },
            )

            # Determine success based on critical issues
            critical_issues = [
                issue
                for issue in validation_issues
                if issue.severity == ValidationSeverity.CRITICAL
            ]
            success = len(critical_issues) == 0

            return IngestionResult(
                success=success,
                data=data if success else None,
                validation_issues=validation_issues,
                data_profile=data_profile,
                processing_time_seconds=processing_time,
                rows_processed=len(data),
                message=(
                    "Data ingestion completed successfully"
                    if success
                    else f"Data ingestion failed with {len(critical_issues)} critical issues"
                ),
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Data ingestion failed: {str(e)}"

            logger.error(error_message)
            audit_logger.log_data_access(
                user_id="system",
                resource=source_path,
                action="data_ingestion",
                success=False,
                details={"error": str(e), "processing_time_seconds": processing_time},
            )

            return IngestionResult(
                success=False,
                data=None,
                validation_issues=[],
                data_profile=None,
                processing_time_seconds=processing_time,
                rows_processed=0,
                message=error_message,
            )

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data quality."""
        validation_issues = self.validator.validate_data(data)
        critical_issues = [
            issue
            for issue in validation_issues
            if issue.severity == ValidationSeverity.CRITICAL
        ]
        return len(critical_issues) == 0

    def _detect_format(self, source_path: str) -> DataFormat:
        """Auto-detect data format from file extension."""
        path = Path(source_path)
        extension = path.suffix.lower()

        format_mapping = {
            ".csv": DataFormat.CSV,
            ".json": DataFormat.JSON,
            ".jsonl": DataFormat.JSON,
            ".parquet": DataFormat.PARQUET,
            ".xlsx": DataFormat.EXCEL,
            ".xls": DataFormat.EXCEL,
        }

        return format_mapping.get(extension, DataFormat.CSV)

    def generate_validation_report(
        self, validation_issues: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not validation_issues:
            return {
                "summary": "No validation issues found",
                "total_issues": 0,
                "issues_by_severity": {},
                "issues_by_type": {},
                "affected_columns": [],
            }

        # Group by severity
        issues_by_severity = {}
        for severity in ValidationSeverity:
            issues_by_severity[severity.value] = len(
                [issue for issue in validation_issues if issue.severity == severity]
            )

        # Group by type
        issues_by_type = {}
        for issue in validation_issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = 0
            issues_by_type[issue.issue_type] += 1

        # Get affected columns
        affected_columns = list(set(issue.column for issue in validation_issues))

        return {
            "summary": f"Found {len(validation_issues)} validation issues",
            "total_issues": len(validation_issues),
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
            "affected_columns": affected_columns,
            "detailed_issues": [
                {
                    "column": issue.column,
                    "type": issue.issue_type,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "count": issue.count,
                    "percentage": round(issue.percentage, 2),
                    "sample_values": issue.sample_values,
                }
                for issue in validation_issues
            ],
        }

    def generate_data_profile_report(self, data_profile: DataProfile) -> Dict[str, Any]:
        """Generate comprehensive data profile report."""
        return {
            "overview": {
                "total_rows": data_profile.total_rows,
                "total_columns": data_profile.total_columns,
                "missing_data_percentage": round(
                    data_profile.missing_data_percentage, 2
                ),
                "duplicate_rows": data_profile.duplicate_rows,
                "data_quality_score": round(data_profile.data_quality_score, 2),
            },
            "column_types": {
                "numeric_columns": data_profile.numeric_columns,
                "categorical_columns": data_profile.categorical_columns,
                "datetime_columns": data_profile.datetime_columns,
            },
            "column_profiles": {
                column: {
                    **profile,
                    "missing_percentage": round(
                        profile.get("missing_percentage", 0), 2
                    ),
                    "unique_percentage": round(profile.get("unique_percentage", 0), 2),
                }
                for column, profile in data_profile.column_profiles.items()
            },
        }


# Factory function for easy access
def create_data_processor() -> DataIngestionProcessor:
    """Create a data ingestion processor instance."""
    return DataIngestionProcessor()


# Utility functions
def ingest_banking_data(file_path: str, **kwargs) -> IngestionResult:
    """Convenience function to ingest banking data."""
    processor = create_data_processor()
    return processor.process(file_path, **kwargs)


def validate_banking_data(data: pd.DataFrame) -> List[ValidationIssue]:
    """Convenience function to validate banking data."""
    validator = BankingDataValidator()
    return validator.validate_data(data)


def profile_banking_data(data: pd.DataFrame) -> DataProfile:
    """Convenience function to profile banking data."""
    profiler = DataProfiler()
    return profiler.profile_data(data)
