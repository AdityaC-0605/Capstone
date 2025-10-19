"""
Carbon Offset Marketplace Integration for Automatic Carbon Neutrality.

This module implements automatic carbon offset purchasing and tracking to achieve
carbon-neutral AI operations. It integrates with carbon offset marketplaces,
calculates required offsets based on AI carbon footprint, and provides
transparent offset tracking and verification.
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from ..core.logging import get_audit_logger, get_logger
    from .carbon_calculator import CarbonCalculator, CarbonFootprint
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger
    from sustainability.carbon_calculator import (
        CarbonCalculator,
        CarbonFootprint,
    )

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class OffsetProjectType(Enum):
    """Types of carbon offset projects."""

    RENEWABLE_ENERGY = "renewable_energy"
    FOREST_CONSERVATION = "forest_conservation"
    REFORESTATION = "reforestation"
    CARBON_CAPTURE = "carbon_capture"
    ENERGY_EFFICIENCY = "energy_efficiency"
    METHANE_CAPTURE = "methane_capture"


class OffsetVerificationStandard(Enum):
    """Carbon offset verification standards."""

    VCS = "vcs"  # Verified Carbon Standard
    GOLD_STANDARD = "gold_standard"
    CARBON_ACTION_RESERVE = "carbon_action_reserve"
    AMERICAN_CARBON_REGISTRY = "american_carbon_registry"
    CLIMATE_ACTION_RESERVE = "climate_action_reserve"


@dataclass
class CarbonOffsetProject:
    """Carbon offset project information."""

    project_id: str
    name: str
    description: str
    project_type: OffsetProjectType
    location: str
    verification_standard: OffsetVerificationStandard

    # Pricing
    price_per_ton_usd: float
    available_credits: int

    # Project details
    start_date: datetime

    # Optional fields with defaults
    min_purchase_kg: float = 100.0  # Minimum purchase in kg
    end_date: Optional[datetime] = None
    estimated_annual_reduction_kg: Optional[float] = None

    # Verification
    verification_id: Optional[str] = None
    verification_date: Optional[datetime] = None
    registry_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "project_type": self.project_type.value,
            "location": self.location,
            "verification_standard": self.verification_standard.value,
            "price_per_ton_usd": self.price_per_ton_usd,
            "available_credits": self.available_credits,
            "min_purchase_kg": self.min_purchase_kg,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "estimated_annual_reduction_kg": self.estimated_annual_reduction_kg,
            "verification_id": self.verification_id,
            "verification_date": (
                self.verification_date.isoformat()
                if self.verification_date
                else None
            ),
            "registry_url": self.registry_url,
        }


@dataclass
class CarbonOffsetPurchase:
    """Carbon offset purchase record."""

    purchase_id: str
    project_id: str
    amount_kg: float
    amount_tonnes: float
    price_per_ton_usd: float
    total_cost_usd: float

    # Purchase details
    purchase_date: datetime
    transaction_id: Optional[str] = None
    payment_method: Optional[str] = None

    # Verification
    offset_certificate_id: Optional[str] = None
    verification_status: str = "pending"
    verification_date: Optional[datetime] = None

    # AI context
    ai_experiment_id: Optional[str] = None
    carbon_footprint_id: Optional[str] = None
    offset_ratio: float = 1.0  # 1.0 = 100% offset, 0.5 = 50% offset

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "purchase_id": self.purchase_id,
            "project_id": self.project_id,
            "amount_kg": self.amount_kg,
            "amount_tonnes": self.amount_tonnes,
            "price_per_ton_usd": self.price_per_ton_usd,
            "total_cost_usd": self.total_cost_usd,
            "purchase_date": self.purchase_date.isoformat(),
            "transaction_id": self.transaction_id,
            "payment_method": self.payment_method,
            "offset_certificate_id": self.offset_certificate_id,
            "verification_status": self.verification_status,
            "verification_date": (
                self.verification_date.isoformat()
                if self.verification_date
                else None
            ),
            "ai_experiment_id": self.ai_experiment_id,
            "carbon_footprint_id": self.carbon_footprint_id,
            "offset_ratio": self.offset_ratio,
        }


@dataclass
class CarbonOffsetConfig:
    """Configuration for carbon offset marketplace integration."""

    # Marketplace settings
    enable_automatic_offsets: bool = True
    auto_offset_threshold_kg: float = (
        0.01  # Auto-purchase offsets above this threshold
    )
    offset_ratio: float = 1.0  # 1.0 = 100% offset, 0.5 = 50% offset

    # Project preferences
    preferred_project_types: List[OffsetProjectType] = field(
        default_factory=lambda: [
            OffsetProjectType.RENEWABLE_ENERGY,
            OffsetProjectType.FOREST_CONSERVATION,
            OffsetProjectType.REFORESTATION,
        ]
    )
    preferred_verification_standards: List[OffsetVerificationStandard] = field(
        default_factory=lambda: [
            OffsetVerificationStandard.VCS,
            OffsetVerificationStandard.GOLD_STANDARD,
        ]
    )

    # Budget constraints
    max_monthly_offset_budget_usd: float = 100.0
    max_single_purchase_usd: float = 50.0

    # API settings
    marketplace_api_url: str = "https://api.carbon-offset-marketplace.com"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    # Payment settings
    payment_method: str = "credit_card"
    billing_address: Optional[Dict[str, str]] = None

    # Verification settings
    require_verification: bool = True
    verification_timeout_days: int = 30

    # Storage
    save_purchases: bool = True
    purchases_file: str = "carbon_offset_purchases.json"


class CarbonOffsetMarketplace:
    """Carbon offset marketplace integration."""

    def __init__(self, config: Optional[CarbonOffsetConfig] = None):
        self.config = config or CarbonOffsetConfig()
        self.carbon_calculator = CarbonCalculator()

        # State
        self.available_projects: List[CarbonOffsetProject] = []
        self.purchase_history: List[CarbonOffsetPurchase] = []
        self.pending_purchases: List[CarbonOffsetPurchase] = []

        # Load existing purchases
        self._load_purchase_history()

        # Initialize marketplace connection
        self._initialize_marketplace()

        logger.info("Carbon offset marketplace initialized")

    def _initialize_marketplace(self):
        """Initialize connection to carbon offset marketplace."""

        try:
            # In a real implementation, this would connect to actual marketplace APIs
            # For demo purposes, we'll create mock projects
            self._load_mock_projects()
            logger.info(
                f"Loaded {len(self.available_projects)} carbon offset projects"
            )

        except Exception as e:
            logger.error(f"Failed to initialize marketplace: {e}")
            self._load_mock_projects()  # Fallback to mock data

    def _load_mock_projects(self):
        """Load mock carbon offset projects for demonstration."""

        mock_projects = [
            CarbonOffsetProject(
                project_id="REN_001",
                name="Wind Farm Development - Texas",
                description="Development of 50MW wind farm in Texas, USA",
                project_type=OffsetProjectType.RENEWABLE_ENERGY,
                location="Texas, USA",
                verification_standard=OffsetVerificationStandard.VCS,
                price_per_ton_usd=12.50,
                available_credits=10000,
                min_purchase_kg=100.0,
                start_date=datetime(2023, 1, 1),
                estimated_annual_reduction_kg=50000.0,
                verification_id="VCS-2023-001",
                verification_date=datetime(2023, 6, 15),
                registry_url="https://registry.verra.org/project/001",
            ),
            CarbonOffsetProject(
                project_id="FOR_002",
                name="Amazon Rainforest Conservation",
                description="Protection of 10,000 hectares of Amazon rainforest",
                project_type=OffsetProjectType.FOREST_CONSERVATION,
                location="Brazil",
                verification_standard=OffsetVerificationStandard.GOLD_STANDARD,
                price_per_ton_usd=15.00,
                available_credits=5000,
                min_purchase_kg=50.0,
                start_date=datetime(2022, 3, 1),
                estimated_annual_reduction_kg=25000.0,
                verification_id="GS-2022-045",
                verification_date=datetime(2022, 9, 20),
                registry_url="https://registry.goldstandard.org/project/045",
            ),
            CarbonOffsetProject(
                project_id="REF_003",
                name="Reforestation Project - Kenya",
                description="Planting 1 million trees in degraded lands in Kenya",
                project_type=OffsetProjectType.REFORESTATION,
                location="Kenya",
                verification_standard=OffsetVerificationStandard.VCS,
                price_per_ton_usd=8.75,
                available_credits=15000,
                min_purchase_kg=25.0,
                start_date=datetime(2023, 5, 1),
                estimated_annual_reduction_kg=30000.0,
                verification_id="VCS-2023-078",
                verification_date=datetime(2023, 11, 10),
                registry_url="https://registry.verra.org/project/078",
            ),
            CarbonOffsetProject(
                project_id="CAP_004",
                name="Direct Air Capture - Iceland",
                description="Direct air capture facility removing CO2 from atmosphere",
                project_type=OffsetProjectType.CARBON_CAPTURE,
                location="Iceland",
                verification_standard=OffsetVerificationStandard.VCS,
                price_per_ton_usd=200.00,
                available_credits=1000,
                min_purchase_kg=10.0,
                start_date=datetime(2024, 1, 1),
                estimated_annual_reduction_kg=5000.0,
                verification_id="VCS-2024-012",
                verification_date=datetime(2024, 3, 1),
                registry_url="https://registry.verra.org/project/012",
            ),
        ]

        self.available_projects = mock_projects

    def _load_purchase_history(self):
        """Load purchase history from file."""

        if not self.config.save_purchases:
            return

        purchases_file = Path(self.config.purchases_file)
        if purchases_file.exists():
            try:
                with open(purchases_file, "r") as f:
                    purchases_data = json.load(f)

                self.purchase_history = []
                for purchase_data in purchases_data:
                    purchase = CarbonOffsetPurchase(
                        purchase_id=purchase_data["purchase_id"],
                        project_id=purchase_data["project_id"],
                        amount_kg=purchase_data["amount_kg"],
                        amount_tonnes=purchase_data["amount_tonnes"],
                        price_per_ton_usd=purchase_data["price_per_ton_usd"],
                        total_cost_usd=purchase_data["total_cost_usd"],
                        purchase_date=datetime.fromisoformat(
                            purchase_data["purchase_date"]
                        ),
                        transaction_id=purchase_data.get("transaction_id"),
                        payment_method=purchase_data.get("payment_method"),
                        offset_certificate_id=purchase_data.get(
                            "offset_certificate_id"
                        ),
                        verification_status=purchase_data.get(
                            "verification_status", "pending"
                        ),
                        verification_date=(
                            datetime.fromisoformat(
                                purchase_data["verification_date"]
                            )
                            if purchase_data.get("verification_date")
                            else None
                        ),
                        ai_experiment_id=purchase_data.get("ai_experiment_id"),
                        carbon_footprint_id=purchase_data.get(
                            "carbon_footprint_id"
                        ),
                        offset_ratio=purchase_data.get("offset_ratio", 1.0),
                    )
                    self.purchase_history.append(purchase)

                logger.info(
                    f"Loaded {len(self.purchase_history)} carbon offset purchases"
                )

            except Exception as e:
                logger.error(f"Failed to load purchase history: {e}")

    def _save_purchase_history(self):
        """Save purchase history to file."""

        if not self.config.save_purchases:
            return

        try:
            purchases_data = [
                purchase.to_dict() for purchase in self.purchase_history
            ]

            purchases_file = Path(self.config.purchases_file)
            with open(purchases_file, "w") as f:
                json.dump(purchases_data, f, indent=2)

            logger.debug("Purchase history saved")

        except Exception as e:
            logger.error(f"Failed to save purchase history: {e}")

    def get_available_projects(
        self,
        project_types: Optional[List[OffsetProjectType]] = None,
        max_price_per_ton: Optional[float] = None,
        min_available_credits: Optional[int] = None,
    ) -> List[CarbonOffsetProject]:
        """Get available carbon offset projects with filters."""

        filtered_projects = self.available_projects.copy()

        # Filter by project type
        if project_types:
            filtered_projects = [
                p for p in filtered_projects if p.project_type in project_types
            ]

        # Filter by price
        if max_price_per_ton:
            filtered_projects = [
                p
                for p in filtered_projects
                if p.price_per_ton_usd <= max_price_per_ton
            ]

        # Filter by available credits
        if min_available_credits:
            filtered_projects = [
                p
                for p in filtered_projects
                if p.available_credits >= min_available_credits
            ]

        # Sort by price (lowest first)
        filtered_projects.sort(key=lambda x: x.price_per_ton_usd)

        return filtered_projects

    def calculate_required_offset(
        self,
        carbon_footprint: CarbonFootprint,
        offset_ratio: Optional[float] = None,
    ) -> float:
        """Calculate required carbon offset amount."""

        if offset_ratio is None:
            offset_ratio = self.config.offset_ratio

        required_offset_kg = carbon_footprint.total_emissions_kg * offset_ratio

        # Round up to nearest 0.1 kg
        required_offset_kg = float(
            Decimal(str(required_offset_kg)).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        )

        return required_offset_kg

    def find_best_offset_project(
        self, required_offset_kg: float
    ) -> Optional[CarbonOffsetProject]:
        """Find the best offset project for the required amount."""

        # Filter projects that can fulfill the requirement
        suitable_projects = [
            p
            for p in self.available_projects
            if (
                p.available_credits
                >= required_offset_kg / 1000  # Convert kg to tonnes
                and p.min_purchase_kg <= required_offset_kg
                and p.project_type in self.config.preferred_project_types
                and p.verification_standard
                in self.config.preferred_verification_standards
            )
        ]

        if not suitable_projects:
            logger.warning(
                f"No suitable offset projects found for {required_offset_kg:.2f} kg"
            )
            return None

        # Sort by price (lowest first)
        suitable_projects.sort(key=lambda x: x.price_per_ton_usd)

        return suitable_projects[0]

    def purchase_carbon_offset(
        self,
        carbon_footprint: CarbonFootprint,
        ai_experiment_id: Optional[str] = None,
        offset_ratio: Optional[float] = None,
    ) -> Optional[CarbonOffsetPurchase]:
        """Purchase carbon offset for a carbon footprint."""

        if not self.config.enable_automatic_offsets:
            logger.info("Automatic carbon offset purchasing is disabled")
            return None

        # Calculate required offset
        required_offset_kg = self.calculate_required_offset(
            carbon_footprint, offset_ratio
        )

        # Check if offset is above threshold
        if required_offset_kg < self.config.auto_offset_threshold_kg:
            logger.info(
                f"Carbon footprint {required_offset_kg:.4f} kg below auto-offset threshold"
            )
            return None

        # Check monthly budget
        monthly_spent = self._calculate_monthly_spending()
        if monthly_spent >= self.config.max_monthly_offset_budget_usd:
            logger.warning("Monthly offset budget exceeded")
            return None

        # Find best project
        project = self.find_best_offset_project(required_offset_kg)
        if not project:
            logger.error("No suitable offset project found")
            return None

        # Calculate cost
        amount_tonnes = required_offset_kg / 1000.0
        total_cost_usd = amount_tonnes * project.price_per_ton_usd

        # Check single purchase limit
        if total_cost_usd > self.config.max_single_purchase_usd:
            logger.warning(
                f"Purchase cost ${total_cost_usd:.2f} exceeds single purchase limit"
            )
            return None

        # Create purchase record
        purchase = CarbonOffsetPurchase(
            purchase_id=f"offset_{int(time.time())}_{hashlib.md5(ai_experiment_id.encode() if ai_experiment_id else b'').hexdigest()[:8]}",
            project_id=project.project_id,
            amount_kg=required_offset_kg,
            amount_tonnes=amount_tonnes,
            price_per_ton_usd=project.price_per_ton_usd,
            total_cost_usd=total_cost_usd,
            purchase_date=datetime.now(),
            ai_experiment_id=ai_experiment_id,
            carbon_footprint_id=carbon_footprint.experiment_id,
            offset_ratio=offset_ratio or self.config.offset_ratio,
        )

        # Simulate purchase process
        success = self._process_purchase(purchase, project)

        if success:
            # Update project available credits
            project.available_credits -= int(amount_tonnes)

            # Add to purchase history
            self.purchase_history.append(purchase)
            self._save_purchase_history()

            # Log audit
            audit_logger.log_model_operation(
                user_id="system",
                model_id="carbon_offset_marketplace",
                operation="purchase_carbon_offset",
                success=True,
                details={
                    "purchase_id": purchase.purchase_id,
                    "project_id": project.project_id,
                    "amount_kg": required_offset_kg,
                    "cost_usd": total_cost_usd,
                    "ai_experiment_id": ai_experiment_id,
                },
            )

            logger.info(
                f"Successfully purchased {required_offset_kg:.2f} kg carbon offset for ${total_cost_usd:.2f}"
            )
            return purchase

        else:
            logger.error("Failed to process carbon offset purchase")
            return None

    def _process_purchase(
        self, purchase: CarbonOffsetPurchase, project: CarbonOffsetProject
    ) -> bool:
        """Process the carbon offset purchase."""

        try:
            # In a real implementation, this would:
            # 1. Process payment through payment gateway
            # 2. Generate offset certificate
            # 3. Update project registry
            # 4. Send confirmation email

            # Simulate processing delay
            time.sleep(0.1)

            # Generate mock transaction ID
            purchase.transaction_id = (
                f"TXN_{int(time.time())}_{purchase.purchase_id[:8]}"
            )
            purchase.payment_method = self.config.payment_method

            # Generate mock offset certificate
            purchase.offset_certificate_id = f"CERT_{purchase.purchase_id}"
            purchase.verification_status = "verified"
            purchase.verification_date = datetime.now()

            logger.info(f"Processed purchase: {purchase.purchase_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process purchase: {e}")
            return False

    def _calculate_monthly_spending(self) -> float:
        """Calculate total spending in current month."""

        current_month = datetime.now().replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        monthly_purchases = [
            p
            for p in self.purchase_history
            if p.purchase_date >= current_month
        ]

        return sum(p.total_cost_usd for p in monthly_purchases)

    def get_offset_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of carbon offsets for a period."""

        cutoff_date = datetime.now() - timedelta(days=days)

        period_purchases = [
            p for p in self.purchase_history if p.purchase_date >= cutoff_date
        ]

        total_offset_kg = sum(p.amount_kg for p in period_purchases)
        total_cost_usd = sum(p.total_cost_usd for p in period_purchases)

        # Group by project type
        project_type_breakdown = {}
        for purchase in period_purchases:
            project = next(
                (
                    p
                    for p in self.available_projects
                    if p.project_id == purchase.project_id
                ),
                None,
            )
            if project:
                project_type = project.project_type.value
                if project_type not in project_type_breakdown:
                    project_type_breakdown[project_type] = {"kg": 0, "cost": 0}
                project_type_breakdown[project_type][
                    "kg"
                ] += purchase.amount_kg
                project_type_breakdown[project_type][
                    "cost"
                ] += purchase.total_cost_usd

        return {
            "period_days": days,
            "total_purchases": len(period_purchases),
            "total_offset_kg": total_offset_kg,
            "total_cost_usd": total_cost_usd,
            "average_cost_per_kg": (
                total_cost_usd / total_offset_kg if total_offset_kg > 0 else 0
            ),
            "project_type_breakdown": project_type_breakdown,
            "purchases": [p.to_dict() for p in period_purchases],
        }

    def verify_offset_certificate(self, certificate_id: str) -> Dict[str, Any]:
        """Verify a carbon offset certificate."""

        # Find purchase by certificate ID
        purchase = next(
            (
                p
                for p in self.purchase_history
                if p.offset_certificate_id == certificate_id
            ),
            None,
        )

        if not purchase:
            return {"valid": False, "error": "Certificate not found"}

        # Find project details
        project = next(
            (
                p
                for p in self.available_projects
                if p.project_id == purchase.project_id
            ),
            None,
        )

        if not project:
            return {"valid": False, "error": "Project not found"}

        return {
            "valid": True,
            "certificate_id": certificate_id,
            "purchase": purchase.to_dict(),
            "project": project.to_dict(),
            "verification_status": purchase.verification_status,
            "verification_date": (
                purchase.verification_date.isoformat()
                if purchase.verification_date
                else None
            ),
        }

    def get_carbon_neutrality_status(self) -> Dict[str, Any]:
        """Get current carbon neutrality status."""

        # Calculate total offsets purchased
        total_offset_kg = sum(p.amount_kg for p in self.purchase_history)

        # Calculate total AI carbon footprint (this would come from carbon calculator)
        # For demo purposes, we'll use a mock value
        total_ai_carbon_kg = 0.5  # Mock value

        # Calculate neutrality ratio
        neutrality_ratio = (
            total_offset_kg / total_ai_carbon_kg
            if total_ai_carbon_kg > 0
            else 0
        )

        is_carbon_neutral = neutrality_ratio >= 1.0

        return {
            "is_carbon_neutral": is_carbon_neutral,
            "neutrality_ratio": neutrality_ratio,
            "total_ai_carbon_kg": total_ai_carbon_kg,
            "total_offset_kg": total_offset_kg,
            "excess_offset_kg": max(0, total_offset_kg - total_ai_carbon_kg),
            "deficit_offset_kg": max(0, total_ai_carbon_kg - total_offset_kg),
            "total_offset_cost_usd": sum(
                p.total_cost_usd for p in self.purchase_history
            ),
        }


# Utility functions


def create_carbon_offset_marketplace(
    config: Optional[CarbonOffsetConfig] = None,
) -> CarbonOffsetMarketplace:
    """Create and configure carbon offset marketplace."""
    return CarbonOffsetMarketplace(config)


def auto_offset_carbon_footprint(
    carbon_footprint: CarbonFootprint,
    ai_experiment_id: Optional[str] = None,
    config: Optional[CarbonOffsetConfig] = None,
) -> Optional[CarbonOffsetPurchase]:
    """Automatically purchase carbon offset for a carbon footprint."""

    marketplace = create_carbon_offset_marketplace(config)
    return marketplace.purchase_carbon_offset(
        carbon_footprint, ai_experiment_id
    )


def get_carbon_neutrality_report(
    config: Optional[CarbonOffsetConfig] = None,
) -> Dict[str, Any]:
    """Get comprehensive carbon neutrality report."""

    marketplace = create_carbon_offset_marketplace(config)

    return {
        "neutrality_status": marketplace.get_carbon_neutrality_status(),
        "offset_summary_30d": marketplace.get_offset_summary(30),
        "offset_summary_90d": marketplace.get_offset_summary(90),
        "available_projects": len(marketplace.available_projects),
        "total_purchases": len(marketplace.purchase_history),
    }
