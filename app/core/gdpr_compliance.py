"""
GDPR compliance framework with data retention policies, right-to-be-forgotten,
consent management, and data lineage tracking.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ..core.config import get_config
from ..core.logging import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ConsentType(Enum):
    """Types of consent under GDPR."""

    PROCESSING = "processing"
    MARKETING = "marketing"
    PROFILING = "profiling"
    THIRD_PARTY_SHARING = "third_party_sharing"
    AUTOMATED_DECISION_MAKING = "automated_decision_making"


class ConsentStatus(Enum):
    """Consent status values."""

    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


class DataCategory(Enum):
    """Categories of personal data."""

    BASIC_IDENTITY = "basic_identity"
    CONTACT_INFO = "contact_info"
    FINANCIAL_DATA = "financial_data"
    BEHAVIORAL_DATA = "behavioral_data"
    BIOMETRIC_DATA = "biometric_data"
    SPECIAL_CATEGORY = "special_category"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""

    CREDIT_ASSESSMENT = "credit_assessment"
    FRAUD_PREVENTION = "fraud_prevention"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    MARKETING = "marketing"
    RESEARCH = "research"
    SERVICE_IMPROVEMENT = "service_improvement"


class LegalBasis(Enum):
    """Legal basis for processing under GDPR."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class ConsentRecord:
    """Individual consent record."""

    consent_id: str
    data_subject_id: str
    consent_type: ConsentType
    status: ConsentStatus
    given_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    expires_at: Optional[datetime]
    purpose: ProcessingPurpose
    legal_basis: LegalBasis
    data_categories: Set[DataCategory]
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""

    policy_id: str
    name: str
    data_category: DataCategory
    purpose: ProcessingPurpose
    retention_period_days: int
    legal_basis: LegalBasis
    auto_delete: bool = True
    review_required: bool = False
    exceptions: List[str] = field(default_factory=list)


@dataclass
class DataLineageRecord:
    """Data lineage tracking record."""

    record_id: str
    data_subject_id: str
    source_system: str
    destination_system: str
    data_categories: Set[DataCategory]
    processing_purpose: ProcessingPurpose
    timestamp: datetime
    transformation_applied: Optional[str]
    retention_policy_id: str
    consent_id: Optional[str]


@dataclass
class DeletionRequest:
    """Right to be forgotten deletion request."""

    request_id: str
    data_subject_id: str
    requested_at: datetime
    processed_at: Optional[datetime]
    status: str  # pending, processing, completed, rejected
    reason: Optional[str]
    affected_systems: List[str] = field(default_factory=list)
    verification_required: bool = True
    legal_review_required: bool = False


class ConsentManager:
    """Manages consent records and consent lifecycle."""

    def __init__(self):
        self.config = get_config()
        self.consent_records: Dict[str, ConsentRecord] = {}
        self._load_consent_records()

    def record_consent(
        self,
        data_subject_id: str,
        consent_type: ConsentType,
        purpose: ProcessingPurpose,
        legal_basis: LegalBasis,
        data_categories: Set[DataCategory],
        expires_in_days: Optional[int] = None,
    ) -> str:
        """Record new consent."""
        consent_id = hashlib.sha256(
            f"{data_subject_id}_{consent_type.value}_{purpose.value}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        consent_record = ConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            consent_type=consent_type,
            status=ConsentStatus.GIVEN,
            given_at=datetime.now(),
            withdrawn_at=None,
            expires_at=expires_at,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
        )

        self.consent_records[consent_id] = consent_record
        self._save_consent_records()

        audit_logger.log_security_event(
            event_type="consent_recorded",
            user_id=None,
            severity="INFO",
            details={
                "consent_id": consent_id,
                "data_subject_id": data_subject_id,
                "consent_type": consent_type.value,
                "purpose": purpose.value,
            },
        )

        logger.info(
            f"Consent recorded: {consent_id} for subject {data_subject_id}"
        )
        return consent_id

    def withdraw_consent(self, consent_id: str, data_subject_id: str) -> bool:
        """Withdraw consent."""
        consent_record = self.consent_records.get(consent_id)
        if (
            not consent_record
            or consent_record.data_subject_id != data_subject_id
        ):
            return False

        consent_record.status = ConsentStatus.WITHDRAWN
        consent_record.withdrawn_at = datetime.now()

        self._save_consent_records()

        audit_logger.log_security_event(
            event_type="consent_withdrawn",
            user_id=None,
            severity="INFO",
            details={
                "consent_id": consent_id,
                "data_subject_id": data_subject_id,
            },
        )

        logger.info(f"Consent withdrawn: {consent_id}")
        return True

    def check_consent_validity(
        self,
        data_subject_id: str,
        purpose: ProcessingPurpose,
        data_category: DataCategory,
    ) -> bool:
        """Check if valid consent exists for processing."""
        current_time = datetime.now()

        for consent_record in self.consent_records.values():
            if (
                consent_record.data_subject_id == data_subject_id
                and consent_record.purpose == purpose
                and data_category in consent_record.data_categories
                and consent_record.status == ConsentStatus.GIVEN
                and (
                    consent_record.expires_at is None
                    or consent_record.expires_at > current_time
                )
            ):
                return True

        return False

    def get_subject_consents(
        self, data_subject_id: str
    ) -> List[ConsentRecord]:
        """Get all consent records for a data subject."""
        return [
            record
            for record in self.consent_records.values()
            if record.data_subject_id == data_subject_id
        ]

    def check_expired_consents(self) -> List[str]:
        """Check for expired consents and update status."""
        current_time = datetime.now()
        expired_consents = []

        for consent_id, consent_record in self.consent_records.items():
            if (
                consent_record.expires_at
                and consent_record.expires_at <= current_time
                and consent_record.status == ConsentStatus.GIVEN
            ):

                consent_record.status = ConsentStatus.EXPIRED
                expired_consents.append(consent_id)

        if expired_consents:
            self._save_consent_records()
            logger.info(f"Expired {len(expired_consents)} consent records")

        return expired_consents

    def _load_consent_records(self):
        """Load consent records from storage."""
        consent_file = Path(self.config.keys_path) / "consent_records.json"

        if consent_file.exists():
            try:
                with open(consent_file, "r") as f:
                    consent_data = json.load(f)

                for consent_id, record_data in consent_data.items():
                    consent_record = ConsentRecord(
                        consent_id=record_data["consent_id"],
                        data_subject_id=record_data["data_subject_id"],
                        consent_type=ConsentType(record_data["consent_type"]),
                        status=ConsentStatus(record_data["status"]),
                        given_at=(
                            datetime.fromisoformat(record_data["given_at"])
                            if record_data.get("given_at")
                            else None
                        ),
                        withdrawn_at=(
                            datetime.fromisoformat(record_data["withdrawn_at"])
                            if record_data.get("withdrawn_at")
                            else None
                        ),
                        expires_at=(
                            datetime.fromisoformat(record_data["expires_at"])
                            if record_data.get("expires_at")
                            else None
                        ),
                        purpose=ProcessingPurpose(record_data["purpose"]),
                        legal_basis=LegalBasis(record_data["legal_basis"]),
                        data_categories={
                            DataCategory(cat)
                            for cat in record_data["data_categories"]
                        },
                        version=record_data.get("version", "1.0"),
                        metadata=record_data.get("metadata", {}),
                    )
                    self.consent_records[consent_id] = consent_record

                logger.info(
                    f"Loaded {len(self.consent_records)} consent records"
                )
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")

    def _save_consent_records(self):
        """Save consent records to storage."""
        consent_path = Path(self.config.keys_path)
        consent_path.mkdir(parents=True, exist_ok=True)
        consent_file = consent_path / "consent_records.json"

        consent_data = {}
        for consent_id, record in self.consent_records.items():
            consent_data[consent_id] = {
                "consent_id": record.consent_id,
                "data_subject_id": record.data_subject_id,
                "consent_type": record.consent_type.value,
                "status": record.status.value,
                "given_at": (
                    record.given_at.isoformat() if record.given_at else None
                ),
                "withdrawn_at": (
                    record.withdrawn_at.isoformat()
                    if record.withdrawn_at
                    else None
                ),
                "expires_at": (
                    record.expires_at.isoformat()
                    if record.expires_at
                    else None
                ),
                "purpose": record.purpose.value,
                "legal_basis": record.legal_basis.value,
                "data_categories": [
                    cat.value for cat in record.data_categories
                ],
                "version": record.version,
                "metadata": record.metadata,
            }

        with open(consent_file, "w") as f:
            json.dump(consent_data, f, indent=2)


class DataRetentionManager:
    """Manages data retention policies and automated deletion."""

    def __init__(self):
        self.config = get_config()
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self._initialize_default_policies()
        self._load_retention_policies()

    def _initialize_default_policies(self):
        """Initialize default retention policies."""
        default_policies = [
            DataRetentionPolicy(
                policy_id="credit_assessment_basic",
                name="Credit Assessment - Basic Identity",
                data_category=DataCategory.BASIC_IDENTITY,
                purpose=ProcessingPurpose.CREDIT_ASSESSMENT,
                retention_period_days=2555,  # 7 years
                legal_basis=LegalBasis.LEGAL_OBLIGATION,
                auto_delete=False,  # Regulatory requirement
                review_required=True,
            ),
            DataRetentionPolicy(
                policy_id="marketing_contact",
                name="Marketing - Contact Information",
                data_category=DataCategory.CONTACT_INFO,
                purpose=ProcessingPurpose.MARKETING,
                retention_period_days=1095,  # 3 years
                legal_basis=LegalBasis.CONSENT,
                auto_delete=True,
            ),
            DataRetentionPolicy(
                policy_id="behavioral_research",
                name="Research - Behavioral Data",
                data_category=DataCategory.BEHAVIORAL_DATA,
                purpose=ProcessingPurpose.RESEARCH,
                retention_period_days=1825,  # 5 years
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
                auto_delete=True,
            ),
        ]

        for policy in default_policies:
            self.retention_policies[policy.policy_id] = policy

    def add_retention_policy(self, policy: DataRetentionPolicy) -> str:
        """Add a new retention policy."""
        self.retention_policies[policy.policy_id] = policy
        self._save_retention_policies()

        audit_logger.log_security_event(
            event_type="retention_policy_added",
            user_id=None,
            severity="INFO",
            details={
                "policy_id": policy.policy_id,
                "data_category": policy.data_category.value,
                "retention_days": policy.retention_period_days,
            },
        )

        return policy.policy_id

    def get_retention_policy(
        self, data_category: DataCategory, purpose: ProcessingPurpose
    ) -> Optional[DataRetentionPolicy]:
        """Get applicable retention policy."""
        for policy in self.retention_policies.values():
            if (
                policy.data_category == data_category
                and policy.purpose == purpose
            ):
                return policy
        return None

    def check_data_expiration(
        self, data_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check which data records have expired based on retention policies."""
        expired_records = []
        current_time = datetime.now()

        for record in data_records:
            # Determine applicable policy
            data_category = DataCategory(record.get("data_category"))
            purpose = ProcessingPurpose(record.get("purpose"))
            created_at = datetime.fromisoformat(record.get("created_at"))

            policy = self.get_retention_policy(data_category, purpose)
            if policy:
                expiration_date = created_at + timedelta(
                    days=policy.retention_period_days
                )
                if current_time > expiration_date:
                    expired_records.append(
                        {
                            **record,
                            "policy_id": policy.policy_id,
                            "expired_at": expiration_date.isoformat(),
                            "auto_delete": policy.auto_delete,
                        }
                    )

        return expired_records

    def _load_retention_policies(self):
        """Load retention policies from storage."""
        policies_file = Path(self.config.keys_path) / "retention_policies.json"

        if policies_file.exists():
            try:
                with open(policies_file, "r") as f:
                    policies_data = json.load(f)

                for policy_id, policy_data in policies_data.items():
                    if (
                        policy_id not in self.retention_policies
                    ):  # Don't override defaults
                        policy = DataRetentionPolicy(
                            policy_id=policy_data["policy_id"],
                            name=policy_data["name"],
                            data_category=DataCategory(
                                policy_data["data_category"]
                            ),
                            purpose=ProcessingPurpose(policy_data["purpose"]),
                            retention_period_days=policy_data[
                                "retention_period_days"
                            ],
                            legal_basis=LegalBasis(policy_data["legal_basis"]),
                            auto_delete=policy_data["auto_delete"],
                            review_required=policy_data["review_required"],
                            exceptions=policy_data.get("exceptions", []),
                        )
                        self.retention_policies[policy_id] = policy

                logger.info(
                    f"Loaded {len(self.retention_policies)} retention policies"
                )
            except Exception as e:
                logger.error(f"Failed to load retention policies: {e}")

    def _save_retention_policies(self):
        """Save retention policies to storage."""
        policies_path = Path(self.config.keys_path)
        policies_path.mkdir(parents=True, exist_ok=True)
        policies_file = policies_path / "retention_policies.json"

        policies_data = {}
        for policy_id, policy in self.retention_policies.items():
            policies_data[policy_id] = {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "data_category": policy.data_category.value,
                "purpose": policy.purpose.value,
                "retention_period_days": policy.retention_period_days,
                "legal_basis": policy.legal_basis.value,
                "auto_delete": policy.auto_delete,
                "review_required": policy.review_required,
                "exceptions": policy.exceptions,
            }

        with open(policies_file, "w") as f:
            json.dump(policies_data, f, indent=2)


class DataLineageTracker:
    """Tracks data lineage for compliance audits."""

    def __init__(self):
        self.config = get_config()
        self.lineage_records: List[DataLineageRecord] = []
        self._load_lineage_records()

    def track_data_flow(
        self,
        data_subject_id: str,
        source_system: str,
        destination_system: str,
        data_categories: Set[DataCategory],
        processing_purpose: ProcessingPurpose,
        retention_policy_id: str,
        consent_id: Optional[str] = None,
        transformation_applied: Optional[str] = None,
    ) -> str:
        """Track data flow between systems."""
        record_id = hashlib.sha256(
            f"{data_subject_id}_{source_system}_{destination_system}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        lineage_record = DataLineageRecord(
            record_id=record_id,
            data_subject_id=data_subject_id,
            source_system=source_system,
            destination_system=destination_system,
            data_categories=data_categories,
            processing_purpose=processing_purpose,
            timestamp=datetime.now(),
            transformation_applied=transformation_applied,
            retention_policy_id=retention_policy_id,
            consent_id=consent_id,
        )

        self.lineage_records.append(lineage_record)
        self._save_lineage_records()

        audit_logger.log_data_access(
            user_id="system",
            resource=f"{source_system}->{destination_system}",
            action="data_transfer",
            success=True,
            details={
                "record_id": record_id,
                "data_subject_id": data_subject_id,
                "data_categories": [cat.value for cat in data_categories],
                "purpose": processing_purpose.value,
            },
        )

        return record_id

    def get_subject_lineage(
        self, data_subject_id: str
    ) -> List[DataLineageRecord]:
        """Get complete data lineage for a subject."""
        return [
            record
            for record in self.lineage_records
            if record.data_subject_id == data_subject_id
        ]

    def get_system_lineage(self, system_name: str) -> List[DataLineageRecord]:
        """Get data lineage involving a specific system."""
        return [
            record
            for record in self.lineage_records
            if record.source_system == system_name
            or record.destination_system == system_name
        ]

    def generate_lineage_report(self, data_subject_id: str) -> Dict[str, Any]:
        """Generate comprehensive lineage report for a data subject."""
        subject_records = self.get_subject_lineage(data_subject_id)

        systems_involved = set()
        data_categories = set()
        purposes = set()

        for record in subject_records:
            systems_involved.add(record.source_system)
            systems_involved.add(record.destination_system)
            data_categories.update(record.data_categories)
            purposes.add(record.processing_purpose)

        return {
            "data_subject_id": data_subject_id,
            "total_transfers": len(subject_records),
            "systems_involved": list(systems_involved),
            "data_categories": [cat.value for cat in data_categories],
            "processing_purposes": [purpose.value for purpose in purposes],
            "first_transfer": (
                min(record.timestamp for record in subject_records).isoformat()
                if subject_records
                else None
            ),
            "last_transfer": (
                max(record.timestamp for record in subject_records).isoformat()
                if subject_records
                else None
            ),
            "transfers": [
                {
                    "record_id": record.record_id,
                    "source": record.source_system,
                    "destination": record.destination_system,
                    "timestamp": record.timestamp.isoformat(),
                    "purpose": record.processing_purpose.value,
                    "transformation": record.transformation_applied,
                }
                for record in sorted(
                    subject_records, key=lambda x: x.timestamp
                )
            ],
        }

    def _load_lineage_records(self):
        """Load lineage records from storage."""
        lineage_file = Path(self.config.logs_path) / "data_lineage.json"

        if lineage_file.exists():
            try:
                with open(lineage_file, "r") as f:
                    lineage_data = json.load(f)

                for record_data in lineage_data:
                    lineage_record = DataLineageRecord(
                        record_id=record_data["record_id"],
                        data_subject_id=record_data["data_subject_id"],
                        source_system=record_data["source_system"],
                        destination_system=record_data["destination_system"],
                        data_categories={
                            DataCategory(cat)
                            for cat in record_data["data_categories"]
                        },
                        processing_purpose=ProcessingPurpose(
                            record_data["processing_purpose"]
                        ),
                        timestamp=datetime.fromisoformat(
                            record_data["timestamp"]
                        ),
                        transformation_applied=record_data.get(
                            "transformation_applied"
                        ),
                        retention_policy_id=record_data["retention_policy_id"],
                        consent_id=record_data.get("consent_id"),
                    )
                    self.lineage_records.append(lineage_record)

                logger.info(
                    f"Loaded {len(self.lineage_records)} lineage records"
                )
            except Exception as e:
                logger.error(f"Failed to load lineage records: {e}")

    def _save_lineage_records(self):
        """Save lineage records to storage."""
        lineage_path = Path(self.config.logs_path)
        lineage_path.mkdir(parents=True, exist_ok=True)
        lineage_file = lineage_path / "data_lineage.json"

        lineage_data = []
        for record in self.lineage_records:
            lineage_data.append(
                {
                    "record_id": record.record_id,
                    "data_subject_id": record.data_subject_id,
                    "source_system": record.source_system,
                    "destination_system": record.destination_system,
                    "data_categories": [
                        cat.value for cat in record.data_categories
                    ],
                    "processing_purpose": record.processing_purpose.value,
                    "timestamp": record.timestamp.isoformat(),
                    "transformation_applied": record.transformation_applied,
                    "retention_policy_id": record.retention_policy_id,
                    "consent_id": record.consent_id,
                }
            )

        with open(lineage_file, "w") as f:
            json.dump(lineage_data, f, indent=2)


class RightToBeForgottenManager:
    """Manages right to be forgotten (data deletion) requests."""

    def __init__(self, lineage_tracker: DataLineageTracker):
        self.config = get_config()
        self.lineage_tracker = lineage_tracker
        self.deletion_requests: Dict[str, DeletionRequest] = {}
        self._load_deletion_requests()

    def submit_deletion_request(
        self,
        data_subject_id: str,
        verification_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a right to be forgotten request."""
        request_id = hashlib.sha256(
            f"{data_subject_id}_deletion_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Get all systems that have processed this subject's data
        lineage_records = self.lineage_tracker.get_subject_lineage(
            data_subject_id
        )
        affected_systems = set()
        for record in lineage_records:
            affected_systems.add(record.source_system)
            affected_systems.add(record.destination_system)

        deletion_request = DeletionRequest(
            request_id=request_id,
            data_subject_id=data_subject_id,
            requested_at=datetime.now(),
            processed_at=None,
            status="pending",
            reason=None,
            affected_systems=list(affected_systems),
            verification_required=True,
            legal_review_required=len(affected_systems)
            > 5,  # Complex cases need review
        )

        self.deletion_requests[request_id] = deletion_request
        self._save_deletion_requests()

        audit_logger.log_security_event(
            event_type="deletion_request_submitted",
            user_id=None,
            severity="INFO",
            details={
                "request_id": request_id,
                "data_subject_id": data_subject_id,
                "affected_systems": list(affected_systems),
            },
        )

        logger.info(
            f"Deletion request submitted: {request_id} for subject {data_subject_id}"
        )
        return request_id

    def process_deletion_request(
        self, request_id: str, approved: bool, reason: Optional[str] = None
    ) -> bool:
        """Process a deletion request."""
        deletion_request = self.deletion_requests.get(request_id)
        if not deletion_request:
            return False

        if approved:
            deletion_request.status = "processing"
            deletion_request.processed_at = datetime.now()

            # Execute deletion across all affected systems
            deletion_results = self._execute_deletion(deletion_request)

            if all(deletion_results.values()):
                deletion_request.status = "completed"
                audit_logger.log_security_event(
                    event_type="deletion_request_completed",
                    user_id=None,
                    severity="INFO",
                    details={
                        "request_id": request_id,
                        "data_subject_id": deletion_request.data_subject_id,
                        "systems_processed": list(deletion_results.keys()),
                    },
                )
            else:
                deletion_request.status = "partial"
                logger.warning(
                    f"Partial deletion for request {request_id}: {deletion_results}"
                )
        else:
            deletion_request.status = "rejected"
            deletion_request.reason = reason
            deletion_request.processed_at = datetime.now()

            audit_logger.log_security_event(
                event_type="deletion_request_rejected",
                user_id=None,
                severity="INFO",
                details={
                    "request_id": request_id,
                    "data_subject_id": deletion_request.data_subject_id,
                    "reason": reason,
                },
            )

        self._save_deletion_requests()
        return True

    def _execute_deletion(
        self, deletion_request: DeletionRequest
    ) -> Dict[str, bool]:
        """Execute deletion across all affected systems."""
        deletion_results = {}

        for system in deletion_request.affected_systems:
            try:
                # This would integrate with actual system deletion APIs
                # For now, we simulate the deletion
                success = self._delete_from_system(
                    system, deletion_request.data_subject_id
                )
                deletion_results[system] = success

                if success:
                    logger.info(
                        f"Deleted data for {deletion_request.data_subject_id} from {system}"
                    )
                else:
                    logger.error(
                        f"Failed to delete data for {deletion_request.data_subject_id} from {system}"
                    )

            except Exception as e:
                logger.error(f"Error deleting from {system}: {e}")
                deletion_results[system] = False

        return deletion_results

    def _delete_from_system(self, system: str, data_subject_id: str) -> bool:
        """Delete data from a specific system (placeholder implementation)."""
        # This would be implemented to integrate with actual systems
        # For now, return True to simulate successful deletion
        return True

    def get_deletion_status(
        self, request_id: str
    ) -> Optional[DeletionRequest]:
        """Get the status of a deletion request."""
        return self.deletion_requests.get(request_id)

    def _load_deletion_requests(self):
        """Load deletion requests from storage."""
        requests_file = Path(self.config.keys_path) / "deletion_requests.json"

        if requests_file.exists():
            try:
                with open(requests_file, "r") as f:
                    requests_data = json.load(f)

                for request_id, request_data in requests_data.items():
                    deletion_request = DeletionRequest(
                        request_id=request_data["request_id"],
                        data_subject_id=request_data["data_subject_id"],
                        requested_at=datetime.fromisoformat(
                            request_data["requested_at"]
                        ),
                        processed_at=(
                            datetime.fromisoformat(
                                request_data["processed_at"]
                            )
                            if request_data.get("processed_at")
                            else None
                        ),
                        status=request_data["status"],
                        reason=request_data.get("reason"),
                        affected_systems=request_data["affected_systems"],
                        verification_required=request_data[
                            "verification_required"
                        ],
                        legal_review_required=request_data[
                            "legal_review_required"
                        ],
                    )
                    self.deletion_requests[request_id] = deletion_request

                logger.info(
                    f"Loaded {len(self.deletion_requests)} deletion requests"
                )
            except Exception as e:
                logger.error(f"Failed to load deletion requests: {e}")

    def _save_deletion_requests(self):
        """Save deletion requests to storage."""
        requests_path = Path(self.config.keys_path)
        requests_path.mkdir(parents=True, exist_ok=True)
        requests_file = requests_path / "deletion_requests.json"

        requests_data = {}
        for request_id, request in self.deletion_requests.items():
            requests_data[request_id] = {
                "request_id": request.request_id,
                "data_subject_id": request.data_subject_id,
                "requested_at": request.requested_at.isoformat(),
                "processed_at": (
                    request.processed_at.isoformat()
                    if request.processed_at
                    else None
                ),
                "status": request.status,
                "reason": request.reason,
                "affected_systems": request.affected_systems,
                "verification_required": request.verification_required,
                "legal_review_required": request.legal_review_required,
            }

        with open(requests_file, "w") as f:
            json.dump(requests_data, f, indent=2)


class GDPRComplianceManager:
    """Main GDPR compliance orchestrator."""

    def __init__(self):
        self.consent_manager = ConsentManager()
        self.retention_manager = DataRetentionManager()
        self.lineage_tracker = DataLineageTracker()
        self.deletion_manager = RightToBeForgottenManager(self.lineage_tracker)

    def process_data_subject_request(
        self, request_type: str, data_subject_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Process various data subject requests under GDPR."""
        if request_type == "access":
            return self._handle_access_request(data_subject_id)
        elif request_type == "portability":
            return self._handle_portability_request(data_subject_id)
        elif request_type == "rectification":
            return self._handle_rectification_request(
                data_subject_id, **kwargs
            )
        elif request_type == "deletion":
            return self._handle_deletion_request(data_subject_id, **kwargs)
        else:
            raise ValueError(f"Unknown request type: {request_type}")

    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data subject access request (Article 15)."""
        consents = self.consent_manager.get_subject_consents(data_subject_id)
        lineage = self.lineage_tracker.get_subject_lineage(data_subject_id)

        return {
            "request_type": "access",
            "data_subject_id": data_subject_id,
            "consents": [asdict(consent) for consent in consents],
            "data_processing_activities": [
                asdict(record) for record in lineage
            ],
            "lineage_report": self.lineage_tracker.generate_lineage_report(
                data_subject_id
            ),
        }

    def _handle_portability_request(
        self, data_subject_id: str
    ) -> Dict[str, Any]:
        """Handle data portability request (Article 20)."""
        # This would extract and format data for portability
        return {
            "request_type": "portability",
            "data_subject_id": data_subject_id,
            "status": "processing",
            "message": "Data portability request is being processed",
        }

    def _handle_rectification_request(
        self, data_subject_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Handle data rectification request (Article 16)."""
        # This would handle data correction requests
        return {
            "request_type": "rectification",
            "data_subject_id": data_subject_id,
            "status": "processing",
            "message": "Data rectification request is being processed",
        }

    def _handle_deletion_request(
        self, data_subject_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Handle right to be forgotten request (Article 17)."""
        request_id = self.deletion_manager.submit_deletion_request(
            data_subject_id
        )

        return {
            "request_type": "deletion",
            "data_subject_id": data_subject_id,
            "request_id": request_id,
            "status": "submitted",
            "message": "Deletion request has been submitted for review",
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive GDPR compliance report."""
        # Check expired consents
        expired_consents = self.consent_manager.check_expired_consents()

        # Get pending deletion requests
        pending_deletions = [
            req
            for req in self.deletion_manager.deletion_requests.values()
            if req.status == "pending"
        ]

        return {
            "report_generated_at": datetime.now().isoformat(),
            "consent_summary": {
                "total_consents": len(self.consent_manager.consent_records),
                "expired_consents": len(expired_consents),
                "active_consents": len(
                    [
                        c
                        for c in self.consent_manager.consent_records.values()
                        if c.status == ConsentStatus.GIVEN
                    ]
                ),
            },
            "deletion_summary": {
                "total_requests": len(self.deletion_manager.deletion_requests),
                "pending_requests": len(pending_deletions),
                "completed_requests": len(
                    [
                        r
                        for r in self.deletion_manager.deletion_requests.values()
                        if r.status == "completed"
                    ]
                ),
            },
            "data_lineage_summary": {
                "total_transfers": len(self.lineage_tracker.lineage_records),
                "unique_subjects": len(
                    set(
                        record.data_subject_id
                        for record in self.lineage_tracker.lineage_records
                    )
                ),
            },
            "retention_policies": len(
                self.retention_manager.retention_policies
            ),
        }
