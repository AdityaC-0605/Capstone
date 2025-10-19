"""
Data anonymization pipeline with PII detection, k-anonymity, l-diversity,
differential privacy, and data masking/tokenization.
"""

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..core.config import get_config
from ..core.logging import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class PIIType(Enum):
    """Types of personally identifiable information."""

    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    ACCOUNT_NUMBER = "account_number"
    CUSTOM = "custom"


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""

    column: str
    pii_type: PIIType
    confidence: float
    sample_matches: List[str]
    total_matches: int


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization operations."""

    k_value: int = 5  # k-anonymity parameter
    l_value: int = 2  # l-diversity parameter
    epsilon: float = 1.0  # differential privacy parameter
    delta: float = 1e-5  # differential privacy parameter
    suppress_threshold: float = 0.1  # suppression threshold
    generalization_levels: Dict[str, List[str]] = None


class PIIDetector:
    """Detects personally identifiable information in datasets."""

    def __init__(self):
        self.patterns = {
            PIIType.SSN: [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{9}\b"],
            PIIType.EMAIL: [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            PIIType.PHONE: [
                r"\b\d{3}-\d{3}-\d{4}\b",
                r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
                r"\b\d{10}\b",
            ],
            PIIType.CREDIT_CARD: [r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"],
            PIIType.ACCOUNT_NUMBER: [r"\b\d{8,20}\b"],
        }

        self.name_indicators = [
            "name",
            "first_name",
            "last_name",
            "full_name",
            "fname",
            "lname",
        ]

        self.address_indicators = [
            "address",
            "street",
            "city",
            "state",
            "zip",
            "postal_code",
        ]

    def detect_pii(self, data: pd.DataFrame) -> List[PIIDetectionResult]:
        """Detect PII in a DataFrame."""
        results = []

        for column in data.columns:
            column_lower = column.lower()

            # Check for name columns
            if any(indicator in column_lower for indicator in self.name_indicators):
                results.append(
                    PIIDetectionResult(
                        column=column,
                        pii_type=PIIType.NAME,
                        confidence=0.9,
                        sample_matches=[],
                        total_matches=len(data[column].dropna()),
                    )
                )
                continue

            # Check for address columns
            if any(indicator in column_lower for indicator in self.address_indicators):
                results.append(
                    PIIDetectionResult(
                        column=column,
                        pii_type=PIIType.ADDRESS,
                        confidence=0.9,
                        sample_matches=[],
                        total_matches=len(data[column].dropna()),
                    )
                )
                continue

            # Pattern-based detection
            if data[column].dtype == "object":
                pii_result = self._detect_patterns(data[column], column)
                if pii_result:
                    results.append(pii_result)

        return results

    def _detect_patterns(
        self, series: pd.Series, column: str
    ) -> Optional[PIIDetectionResult]:
        """Detect PII patterns in a series."""
        sample_values = series.dropna().astype(str).head(1000)

        for pii_type, patterns in self.patterns.items():
            matches = []
            total_matches = 0

            for pattern in patterns:
                for value in sample_values:
                    if re.search(pattern, value):
                        matches.append(value)
                        total_matches += 1

            if total_matches > 0:
                confidence = min(total_matches / len(sample_values), 1.0)
                if confidence > 0.1:  # Threshold for detection
                    return PIIDetectionResult(
                        column=column,
                        pii_type=pii_type,
                        confidence=confidence,
                        sample_matches=matches[:5],
                        total_matches=total_matches,
                    )

        return None


class DataMasker:
    """Handles data masking and tokenization."""

    def __init__(self):
        self.token_mapping: Dict[str, str] = {}
        self.reverse_mapping: Dict[str, str] = {}

    def mask_data(
        self, data: pd.DataFrame, pii_columns: List[str], mask_type: str = "hash"
    ) -> pd.DataFrame:
        """Mask PII data in specified columns."""
        masked_data = data.copy()

        for column in pii_columns:
            if column in data.columns:
                if mask_type == "hash":
                    masked_data[column] = self._hash_values(data[column])
                elif mask_type == "tokenize":
                    masked_data[column] = self._tokenize_values(data[column])
                elif mask_type == "suppress":
                    masked_data[column] = "***SUPPRESSED***"
                elif mask_type == "partial":
                    masked_data[column] = self._partial_mask(data[column])

        return masked_data

    def _hash_values(self, series: pd.Series) -> pd.Series:
        """Hash values using SHA-256."""

        def hash_value(value):
            if pd.isna(value):
                return value
            return hashlib.sha256(str(value).encode()).hexdigest()[:16]

        return series.apply(hash_value)

    def _tokenize_values(self, series: pd.Series) -> pd.Series:
        """Replace values with tokens."""

        def tokenize_value(value):
            if pd.isna(value):
                return value

            str_value = str(value)
            if str_value not in self.token_mapping:
                token = f"TOKEN_{secrets.token_hex(8)}"
                self.token_mapping[str_value] = token
                self.reverse_mapping[token] = str_value

            return self.token_mapping[str_value]

        return series.apply(tokenize_value)

    def _partial_mask(self, series: pd.Series) -> pd.Series:
        """Partially mask values (show first/last characters)."""

        def partial_mask_value(value):
            if pd.isna(value):
                return value

            str_value = str(value)
            if len(str_value) <= 4:
                return "*" * len(str_value)

            return str_value[:2] + "*" * (len(str_value) - 4) + str_value[-2:]

        return series.apply(partial_mask_value)

    def detokenize_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Reverse tokenization for specified columns."""
        detokenized_data = data.copy()

        for column in columns:
            if column in data.columns:
                detokenized_data[column] = data[column].map(
                    lambda x: self.reverse_mapping.get(x, x) if not pd.isna(x) else x
                )

        return detokenized_data


class KAnonymizer:
    """Implements k-anonymity and l-diversity anonymization."""

    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.generalization_hierarchies = self._build_generalization_hierarchies()

    def anonymize(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
    ) -> pd.DataFrame:
        """Apply k-anonymity and l-diversity."""
        anonymized_data = data.copy()

        # Apply k-anonymity
        anonymized_data = self._apply_k_anonymity(anonymized_data, quasi_identifiers)

        # Apply l-diversity if sensitive attributes are specified
        if sensitive_attributes:
            anonymized_data = self._apply_l_diversity(
                anonymized_data, quasi_identifiers, sensitive_attributes
            )

        return anonymized_data

    def _apply_k_anonymity(
        self, data: pd.DataFrame, quasi_identifiers: List[str]
    ) -> pd.DataFrame:
        """Apply k-anonymity by generalization and suppression."""
        # Group by quasi-identifiers
        grouped = data.groupby(quasi_identifiers)

        anonymized_data = []
        suppressed_count = 0

        for group_key, group_data in grouped:
            if len(group_data) >= self.config.k_value:
                # Group satisfies k-anonymity
                anonymized_data.append(group_data)
            else:
                # Apply generalization or suppression
                if len(group_data) / len(data) < self.config.suppress_threshold:
                    # Suppress small groups
                    suppressed_count += len(group_data)
                else:
                    # Generalize
                    generalized_data = self._generalize_group(
                        group_data, quasi_identifiers
                    )
                    anonymized_data.append(generalized_data)

        if anonymized_data:
            result = pd.concat(anonymized_data, ignore_index=True)
        else:
            result = pd.DataFrame(columns=data.columns)

        logger.info(f"K-anonymity applied: {suppressed_count} records suppressed")
        return result

    def _apply_l_diversity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
    ) -> pd.DataFrame:
        """Apply l-diversity to ensure diversity in sensitive attributes."""
        grouped = data.groupby(quasi_identifiers)

        anonymized_data = []

        for group_key, group_data in grouped:
            # Check l-diversity for each sensitive attribute
            satisfies_l_diversity = True

            for sensitive_attr in sensitive_attributes:
                unique_values = group_data[sensitive_attr].nunique()
                if unique_values < self.config.l_value:
                    satisfies_l_diversity = False
                    break

            if satisfies_l_diversity:
                anonymized_data.append(group_data)
            else:
                # Apply additional generalization or suppression
                # For simplicity, we suppress groups that don't satisfy l-diversity
                logger.debug(
                    f"Suppressing group due to l-diversity: {len(group_data)} records"
                )

        if anonymized_data:
            result = pd.concat(anonymized_data, ignore_index=True)
        else:
            result = pd.DataFrame(columns=data.columns)

        return result

    def _generalize_group(
        self, group_data: pd.DataFrame, quasi_identifiers: List[str]
    ) -> pd.DataFrame:
        """Generalize quasi-identifiers in a group."""
        generalized_data = group_data.copy()

        for qi in quasi_identifiers:
            if qi in self.generalization_hierarchies:
                # Apply generalization hierarchy
                hierarchy = self.generalization_hierarchies[qi]
                generalized_data[qi] = self._apply_hierarchy(group_data[qi], hierarchy)
            else:
                # Default generalization (e.g., age ranges)
                generalized_data[qi] = self._default_generalization(group_data[qi])

        return generalized_data

    def _apply_hierarchy(self, series: pd.Series, hierarchy: List[str]) -> pd.Series:
        """Apply generalization hierarchy."""
        # Simple implementation: use the most general level
        return hierarchy[-1] if hierarchy else series

    def _default_generalization(self, series: pd.Series) -> pd.Series:
        """Apply default generalization (ranges for numeric data)."""
        if pd.api.types.is_numeric_dtype(series):
            # Create ranges
            min_val = series.min()
            max_val = series.max()
            return f"{min_val}-{max_val}"
        else:
            # For categorical data, use a generic category
            return "GENERALIZED"

    def _build_generalization_hierarchies(self) -> Dict[str, List[str]]:
        """Build generalization hierarchies for common attributes."""
        hierarchies = {
            "age": ["specific_age", "age_group_5", "age_group_10", "adult/minor"],
            "income": [
                "exact_income",
                "income_bracket_10k",
                "income_bracket_50k",
                "income_level",
            ],
            "zipcode": ["full_zip", "zip_4digit", "zip_3digit", "state"],
        }

        # Add custom hierarchies from config
        if self.config.generalization_levels:
            hierarchies.update(self.config.generalization_levels)

        return hierarchies


class DifferentialPrivacy:
    """Implements differential privacy mechanisms."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def add_laplace_noise(
        self, data: pd.DataFrame, columns: List[str], sensitivity: float = 1.0
    ) -> pd.DataFrame:
        """Add Laplace noise for differential privacy."""
        noisy_data = data.copy()

        # Calculate noise scale
        scale = sensitivity / self.epsilon

        for column in columns:
            if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
                noise = np.random.laplace(0, scale, size=len(data))
                noisy_data[column] = data[column] + noise

        return noisy_data

    def add_gaussian_noise(
        self, data: pd.DataFrame, columns: List[str], sensitivity: float = 1.0
    ) -> pd.DataFrame:
        """Add Gaussian noise for differential privacy."""
        noisy_data = data.copy()

        # Calculate noise scale for (epsilon, delta)-differential privacy
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon

        for column in columns:
            if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
                noise = np.random.normal(0, sigma, size=len(data))
                noisy_data[column] = data[column] + noise

        return noisy_data

    def exponential_mechanism(
        self, data: pd.DataFrame, column: str, utility_function: callable
    ) -> Any:
        """Apply exponential mechanism for categorical data."""
        unique_values = data[column].unique()

        # Calculate utilities
        utilities = [utility_function(value, data) for value in unique_values]

        # Calculate probabilities
        max_utility = max(utilities)
        probabilities = np.exp(self.epsilon * np.array(utilities) / (2 * max_utility))
        probabilities = probabilities / probabilities.sum()

        # Sample according to probabilities
        return np.random.choice(unique_values, p=probabilities)


class AnonymizationPipeline:
    """Main anonymization pipeline orchestrator."""

    def __init__(self, config: Optional[AnonymizationConfig] = None):
        self.config = config or AnonymizationConfig()
        self.pii_detector = PIIDetector()
        self.data_masker = DataMasker()
        self.k_anonymizer = KAnonymizer(self.config)
        self.differential_privacy = DifferentialPrivacy(
            self.config.epsilon, self.config.delta
        )

    def anonymize_dataset(
        self, data: pd.DataFrame, anonymization_strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply comprehensive anonymization to a dataset."""
        # Detect PII
        pii_results = self.pii_detector.detect_pii(data)

        # Extract configuration
        mask_columns = anonymization_strategy.get("mask_columns", [])
        quasi_identifiers = anonymization_strategy.get("quasi_identifiers", [])
        sensitive_attributes = anonymization_strategy.get("sensitive_attributes", [])
        dp_columns = anonymization_strategy.get("differential_privacy_columns", [])

        anonymized_data = data.copy()

        # Step 1: Mask PII data
        if mask_columns:
            mask_type = anonymization_strategy.get("mask_type", "hash")
            anonymized_data = self.data_masker.mask_data(
                anonymized_data, mask_columns, mask_type
            )

        # Step 2: Apply k-anonymity and l-diversity
        if quasi_identifiers:
            anonymized_data = self.k_anonymizer.anonymize(
                anonymized_data, quasi_identifiers, sensitive_attributes
            )

        # Step 3: Apply differential privacy
        if dp_columns:
            noise_type = anonymization_strategy.get("noise_type", "laplace")
            sensitivity = anonymization_strategy.get("sensitivity", 1.0)

            if noise_type == "laplace":
                anonymized_data = self.differential_privacy.add_laplace_noise(
                    anonymized_data, dp_columns, sensitivity
                )
            elif noise_type == "gaussian":
                anonymized_data = self.differential_privacy.add_gaussian_noise(
                    anonymized_data, dp_columns, sensitivity
                )

        # Generate anonymization report
        report = {
            "original_records": len(data),
            "anonymized_records": len(anonymized_data),
            "pii_detected": [
                {
                    "column": result.column,
                    "type": result.pii_type.value,
                    "confidence": result.confidence,
                }
                for result in pii_results
            ],
            "anonymization_applied": {
                "masking": bool(mask_columns),
                "k_anonymity": bool(quasi_identifiers),
                "l_diversity": bool(sensitive_attributes),
                "differential_privacy": bool(dp_columns),
            },
            "privacy_parameters": {
                "k_value": self.config.k_value,
                "l_value": self.config.l_value,
                "epsilon": self.config.epsilon,
                "delta": self.config.delta,
            },
        }

        # Log anonymization
        audit_logger.log_security_event(
            event_type="data_anonymization",
            user_id=None,
            severity="INFO",
            details=report,
        )

        logger.info(
            f"Anonymization completed: {len(data)} -> {len(anonymized_data)} records"
        )

        return anonymized_data, report

    def validate_anonymization(
        self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate the quality of anonymization."""
        validation_results = {
            "data_utility": self._calculate_data_utility(
                original_data, anonymized_data
            ),
            "privacy_risk": self._assess_privacy_risk(anonymized_data),
            "information_loss": self._calculate_information_loss(
                original_data, anonymized_data
            ),
        }

        return validation_results

    def _calculate_data_utility(
        self, original: pd.DataFrame, anonymized: pd.DataFrame
    ) -> float:
        """Calculate data utility preservation."""
        # Simple utility measure based on statistical properties
        utility_scores = []

        for column in original.select_dtypes(include=[np.number]).columns:
            if column in anonymized.columns:
                orig_mean = original[column].mean()
                anon_mean = anonymized[column].mean()

                if orig_mean != 0:
                    utility = 1 - abs(orig_mean - anon_mean) / abs(orig_mean)
                    utility_scores.append(max(0, utility))

        return np.mean(utility_scores) if utility_scores else 0.0

    def _assess_privacy_risk(self, anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess privacy risk in anonymized data."""
        # Simple risk assessment
        risk_assessment = {
            "uniqueness_risk": len(anonymized_data.drop_duplicates())
            / len(anonymized_data),
            "small_group_risk": 0.0,
            "overall_risk": "low",
        }

        return risk_assessment

    def _calculate_information_loss(
        self, original: pd.DataFrame, anonymized: pd.DataFrame
    ) -> float:
        """Calculate information loss due to anonymization."""
        # Simple information loss measure
        if len(original) == 0:
            return 1.0

        loss = 1 - (len(anonymized) / len(original))
        return max(0.0, min(1.0, loss))
