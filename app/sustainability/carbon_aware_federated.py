"""
Carbon-Aware Federated Learning Implementation.

This module implements carbon-aware federated learning that considers carbon intensity
when selecting clients and scheduling training rounds. It integrates with the existing
federated learning infrastructure to optimize for both performance and sustainability.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from ..core.logging import get_audit_logger, get_logger
    from ..federated.federated_client import ClientConfig, FederatedClient
    from ..federated.federated_server import (
        AggregationMethod,
        ClientInfo,
        ClientSelector,
        FederatedConfig,
        FederatedRound,
        FederatedServer,
        ModelUpdate,
    )
    from ..sustainability.carbon_aware_optimizer import (
        CarbonAwareConfig,
        CarbonAwareOptimizer,
    )
    from ..sustainability.carbon_calculator import (
        CarbonCalculator,
        CarbonFootprint,
    )
    from ..sustainability.energy_tracker import EnergyTracker
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger
    from federated.federated_client import ClientConfig, FederatedClient
    from federated.federated_server import (
        AggregationMethod,
        ClientInfo,
        ClientSelector,
        FederatedConfig,
        FederatedRound,
        FederatedServer,
        ModelUpdate,
    )
    from sustainability.carbon_aware_optimizer import (
        CarbonAwareConfig,
        CarbonAwareOptimizer,
    )
    from sustainability.carbon_calculator import (
        CarbonCalculator,
        CarbonFootprint,
    )
    from sustainability.energy_tracker import EnergyTracker

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class CarbonAwareSelectionStrategy(Enum):
    """Carbon-aware client selection strategies."""

    LOWEST_CARBON = (
        "lowest_carbon"  # Select clients with lowest carbon intensity
    )
    CARBON_BUDGET = "carbon_budget"  # Select clients within carbon budget
    ADAPTIVE = "adaptive"  # Adapt selection based on carbon intensity trends
    HYBRID = "hybrid"  # Balance carbon efficiency with performance


@dataclass
class CarbonAwareClientInfo:
    """Enhanced client information with carbon metrics."""

    client_id: str
    region: str
    carbon_intensity_gco2_kwh: float
    renewable_energy_percentage: float
    last_carbon_update: datetime
    carbon_efficiency_score: float = 0.0
    energy_efficiency_score: float = 0.0
    sustainability_rating: str = "B"  # A+, A, B, C, D

    # Performance metrics
    average_training_time: float = 0.0
    average_energy_consumption: float = 0.0
    model_quality_score: float = 0.0

    # Carbon budget tracking
    carbon_budget_used: float = 0.0
    carbon_budget_limit: float = 1.0  # kg CO2e per day

    def is_carbon_budget_available(self) -> bool:
        """Check if client has carbon budget available."""
        return self.carbon_budget_used < self.carbon_budget_limit

    def get_carbon_efficiency_rating(self) -> str:
        """Get carbon efficiency rating based on metrics."""
        if self.carbon_efficiency_score >= 0.9:
            return "A+"
        elif self.carbon_efficiency_score >= 0.8:
            return "A"
        elif self.carbon_efficiency_score >= 0.7:
            return "B"
        elif self.carbon_efficiency_score >= 0.6:
            return "C"
        else:
            return "D"


@dataclass
class CarbonAwareFederatedConfig(FederatedConfig):
    """Configuration for carbon-aware federated learning."""

    # Carbon-aware settings
    enable_carbon_aware_selection: bool = True
    carbon_selection_strategy: CarbonAwareSelectionStrategy = (
        CarbonAwareSelectionStrategy.HYBRID
    )
    max_carbon_intensity_threshold: float = 400.0  # gCO2/kWh
    min_renewable_energy_percentage: float = 30.0  # %
    carbon_budget_per_round: float = 0.1  # kg CO2e per round

    # Carbon intensity API settings
    carbon_intensity_api_url: str = (
        "https://api.carbonintensity.org.uk/regional"
    )
    carbon_intensity_update_interval: int = 300  # seconds
    enable_real_time_carbon: bool = True

    # Sustainability optimization
    sustainability_weight: float = (
        0.3  # Weight for sustainability in client selection
    )
    performance_weight: float = (
        0.7  # Weight for performance in client selection
    )
    carbon_efficiency_threshold: float = 0.7

    # Carbon offset integration
    enable_automatic_carbon_offset: bool = True
    carbon_offset_threshold: float = 0.05  # kg CO2e
    carbon_offset_provider: str = "automatic"

    # Monitoring and reporting
    enable_carbon_monitoring: bool = True
    carbon_reporting_interval: int = 3600  # seconds
    save_carbon_reports: bool = True


class CarbonIntensityAPI:
    """Interface for real-time carbon intensity data."""

    def __init__(
        self, api_url: str = "https://api.carbonintensity.org.uk/regional"
    ):
        self.api_url = api_url
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 300  # 5 minutes

    async def get_carbon_intensity(self, region: str) -> Dict[str, Any]:
        """
        Get current carbon intensity for a region.

        Args:
            region: Region identifier (e.g., "US-CA", "EU-DE")

        Returns:
            Dictionary with carbon intensity data
        """
        try:
            # Check cache first
            if region in self.cache:
                cached_data = self.cache[region]
                if (
                    time.time() - cached_data["timestamp"]
                    < self.cache_duration
                ):
                    return cached_data["data"]

            # For demo purposes, simulate API call
            # In production, this would make actual API calls
            carbon_intensity = self._simulate_carbon_intensity(region)

            # Cache the result
            self.cache[region] = {
                "data": carbon_intensity,
                "timestamp": time.time(),
            }

            return carbon_intensity

        except Exception as e:
            logger.error(f"Failed to get carbon intensity for {region}: {e}")
            # Return default values
            return {
                "carbon_intensity": 300.0,  # gCO2/kWh
                "renewable_percentage": 25.0,
                "timestamp": datetime.now().isoformat(),
                "region": region,
            }

    def _simulate_carbon_intensity(self, region: str) -> Dict[str, Any]:
        """Simulate carbon intensity data for demo purposes."""
        # Simulate different carbon intensities by region
        region_intensities = {
            "US-CA": {"carbon_intensity": 180.0, "renewable_percentage": 45.0},
            "US-TX": {"carbon_intensity": 420.0, "renewable_percentage": 20.0},
            "EU-DE": {"carbon_intensity": 350.0, "renewable_percentage": 35.0},
            "EU-FR": {"carbon_intensity": 80.0, "renewable_percentage": 70.0},
            "ASIA-CN": {
                "carbon_intensity": 580.0,
                "renewable_percentage": 15.0,
            },
            "DEFAULT": {
                "carbon_intensity": 300.0,
                "renewable_percentage": 25.0,
            },
        }

        # Add some randomness to simulate real-time variations
        base_data = region_intensities.get(
            region, region_intensities["DEFAULT"]
        )
        variation = random.uniform(0.8, 1.2)

        return {
            "carbon_intensity": base_data["carbon_intensity"] * variation,
            "renewable_percentage": base_data["renewable_percentage"]
            * variation,
            "timestamp": datetime.now().isoformat(),
            "region": region,
        }


class CarbonAwareClientSelector(ClientSelector):
    """Enhanced client selector that considers carbon intensity."""

    def __init__(self, config: CarbonAwareFederatedConfig):
        super().__init__(config)
        self.config = config
        self.carbon_api = CarbonIntensityAPI(config.carbon_intensity_api_url)
        self.carbon_calculator = CarbonCalculator()
        self.energy_tracker = EnergyTracker()

        # Carbon-aware client information
        self.carbon_aware_clients: Dict[str, CarbonAwareClientInfo] = {}

        # Start carbon intensity monitoring
        if config.enable_real_time_carbon:
            asyncio.create_task(self._monitor_carbon_intensity())

    async def _monitor_carbon_intensity(self):
        """Monitor carbon intensity for all clients."""
        while True:
            try:
                await asyncio.sleep(
                    self.config.carbon_intensity_update_interval
                )

                for (
                    client_id,
                    carbon_info,
                ) in self.carbon_aware_clients.items():
                    # Update carbon intensity
                    carbon_data = await self.carbon_api.get_carbon_intensity(
                        carbon_info.region
                    )

                    carbon_info.carbon_intensity_gco2_kwh = carbon_data[
                        "carbon_intensity"
                    ]
                    carbon_info.renewable_energy_percentage = carbon_data[
                        "renewable_percentage"
                    ]
                    carbon_info.last_carbon_update = datetime.now()

                    # Update carbon efficiency score
                    carbon_info.carbon_efficiency_score = (
                        self._calculate_carbon_efficiency_score(carbon_info)
                    )
                    carbon_info.sustainability_rating = (
                        carbon_info.get_carbon_efficiency_rating()
                    )

                logger.debug("Updated carbon intensity for all clients")

            except Exception as e:
                logger.error(f"Error monitoring carbon intensity: {e}")

    def _calculate_carbon_efficiency_score(
        self, carbon_info: CarbonAwareClientInfo
    ) -> float:
        """Calculate carbon efficiency score for a client."""
        # Normalize carbon intensity (lower is better)
        carbon_score = max(
            0, 1 - (carbon_info.carbon_intensity_gco2_kwh / 600.0)
        )

        # Normalize renewable energy percentage (higher is better)
        renewable_score = carbon_info.renewable_energy_percentage / 100.0

        # Combine scores
        efficiency_score = carbon_score * 0.6 + renewable_score * 0.4
        return min(1.0, max(0.0, efficiency_score))

    def register_carbon_aware_client(
        self, client_id: str, region: str, carbon_budget_limit: float = 1.0
    ) -> bool:
        """
        Register a client with carbon awareness.

        Args:
            client_id: Client identifier
            region: Client's region
            carbon_budget_limit: Daily carbon budget limit in kg CO2e

        Returns:
            True if successful, False otherwise
        """
        try:
            carbon_info = CarbonAwareClientInfo(
                client_id=client_id,
                region=region,
                carbon_intensity_gco2_kwh=300.0,  # Default value
                renewable_energy_percentage=25.0,  # Default value
                last_carbon_update=datetime.now(),
                carbon_budget_limit=carbon_budget_limit,
            )

            self.carbon_aware_clients[client_id] = carbon_info

            logger.info(
                f"Registered carbon-aware client {client_id} in region {region}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to register carbon-aware client {client_id}: {e}"
            )
            return False

    def select_clients(
        self, available_clients: List[ClientInfo], round_number: int
    ) -> List[ClientInfo]:
        """
        Select clients using carbon-aware strategy.

        Args:
            available_clients: List of available clients
            round_number: Current round number

        Returns:
            Selected clients for training
        """
        if not self.config.enable_carbon_aware_selection:
            # Fall back to standard selection
            return super().select_clients(available_clients, round_number)

        # Filter active clients
        active_clients = [
            client
            for client in available_clients
            if client.is_active()
            and client.status.value in ["connected", "ready"]
        ]

        if len(active_clients) < self.config.min_clients_per_round:
            logger.warning(
                f"Not enough active clients: {len(active_clients)} < {self.config.min_clients_per_round}"
            )
            return []

        # Apply carbon-aware selection strategy
        if (
            self.config.carbon_selection_strategy
            == CarbonAwareSelectionStrategy.LOWEST_CARBON
        ):
            return self._select_lowest_carbon_clients(active_clients)
        elif (
            self.config.carbon_selection_strategy
            == CarbonAwareSelectionStrategy.CARBON_BUDGET
        ):
            return self._select_carbon_budget_clients(active_clients)
        elif (
            self.config.carbon_selection_strategy
            == CarbonAwareSelectionStrategy.ADAPTIVE
        ):
            return self._select_adaptive_clients(active_clients, round_number)
        elif (
            self.config.carbon_selection_strategy
            == CarbonAwareSelectionStrategy.HYBRID
        ):
            return self._select_hybrid_clients(active_clients)
        else:
            logger.warning(
                f"Unknown carbon selection strategy: {self.config.carbon_selection_strategy}"
            )
            return self._select_hybrid_clients(active_clients)

    def _select_lowest_carbon_clients(
        self, clients: List[ClientInfo]
    ) -> List[ClientInfo]:
        """Select clients with lowest carbon intensity."""
        # Score clients based on carbon intensity
        scored_clients = []
        for client in clients:
            if client.client_id in self.carbon_aware_clients:
                carbon_info = self.carbon_aware_clients[client.client_id]
                score = 1.0 / (
                    carbon_info.carbon_intensity_gco2_kwh + 1.0
                )  # Lower carbon = higher score
                scored_clients.append((client, score))
            else:
                # Default score for clients without carbon info
                scored_clients.append((client, 0.5))

        # Sort by score (descending)
        scored_clients.sort(key=lambda x: x[1], reverse=True)

        num_clients = min(len(clients), self.config.max_clients_per_round)
        return [client for client, _ in scored_clients[:num_clients]]

    def _select_carbon_budget_clients(
        self, clients: List[ClientInfo]
    ) -> List[ClientInfo]:
        """Select clients within carbon budget."""
        budget_clients = []

        for client in clients:
            if client.client_id in self.carbon_aware_clients:
                carbon_info = self.carbon_aware_clients[client.client_id]

                # Check if client has carbon budget available
                if carbon_info.is_carbon_budget_available():
                    budget_clients.append(client)

        if len(budget_clients) < self.config.min_clients_per_round:
            logger.warning(
                f"Not enough clients within carbon budget: {len(budget_clients)}"
            )
            # Fall back to lowest carbon selection
            return self._select_lowest_carbon_clients(clients)

        num_clients = min(
            len(budget_clients), self.config.max_clients_per_round
        )
        return budget_clients[:num_clients]

    def _select_adaptive_clients(
        self, clients: List[ClientInfo], round_number: int
    ) -> List[ClientInfo]:
        """Adaptively select clients based on carbon intensity trends."""
        # Analyze carbon intensity trends
        current_avg_carbon = np.mean(
            [
                self.carbon_aware_clients[
                    client.client_id
                ].carbon_intensity_gco2_kwh
                for client in clients
                if client.client_id in self.carbon_aware_clients
            ]
        )

        # Adjust selection strategy based on current carbon intensity
        if current_avg_carbon > self.config.max_carbon_intensity_threshold:
            # High carbon intensity - be more selective
            return self._select_lowest_carbon_clients(clients)
        else:
            # Low carbon intensity - can be more flexible
            return self._select_hybrid_clients(clients)

    def _select_hybrid_clients(
        self, clients: List[ClientInfo]
    ) -> List[ClientInfo]:
        """Select clients balancing carbon efficiency and performance."""
        scored_clients = []

        for client in clients:
            sustainability_score = 0.5  # Default
            performance_score = 0.5  # Default

            # Calculate sustainability score
            if client.client_id in self.carbon_aware_clients:
                carbon_info = self.carbon_aware_clients[client.client_id]
                sustainability_score = carbon_info.carbon_efficiency_score

            # Calculate performance score
            performance_score = (
                client.connection_quality * 0.4
                + (1.0 / (client.average_response_time + 1.0)) * 0.3
                + (client.data_size / 1000.0) * 0.3  # Normalize data size
            )

            # Combine scores with weights
            combined_score = (
                sustainability_score * self.config.sustainability_weight
                + performance_score * self.config.performance_weight
            )

            scored_clients.append((client, combined_score))

        # Sort by combined score (descending)
        scored_clients.sort(key=lambda x: x[1], reverse=True)

        num_clients = min(len(clients), self.config.max_clients_per_round)
        return [client for client, _ in scored_clients[:num_clients]]

    def update_client_carbon_budget(
        self, client_id: str, carbon_used: float
    ) -> bool:
        """
        Update client's carbon budget after training.

        Args:
            client_id: Client identifier
            carbon_used: Carbon used in kg CO2e

        Returns:
            True if successful, False otherwise
        """
        try:
            if client_id in self.carbon_aware_clients:
                carbon_info = self.carbon_aware_clients[client_id]
                carbon_info.carbon_budget_used += carbon_used

                logger.debug(
                    f"Updated carbon budget for client {client_id}: "
                    f"used {carbon_used:.4f} kg CO2e, "
                    f"remaining {carbon_info.carbon_budget_limit - carbon_info.carbon_budget_used:.4f} kg CO2e"
                )
                return True
            else:
                logger.warning(
                    f"Client {client_id} not found in carbon-aware clients"
                )
                return False

        except Exception as e:
            logger.error(
                f"Failed to update carbon budget for client {client_id}: {e}"
            )
            return False

    def get_carbon_aware_client_info(
        self, client_id: str
    ) -> Optional[CarbonAwareClientInfo]:
        """Get carbon-aware information for a client."""
        return self.carbon_aware_clients.get(client_id)

    def get_carbon_summary(self) -> Dict[str, Any]:
        """Get summary of carbon-aware client information."""
        if not self.carbon_aware_clients:
            return {"message": "No carbon-aware clients registered"}

        total_clients = len(self.carbon_aware_clients)
        avg_carbon_intensity = np.mean(
            [
                info.carbon_intensity_gco2_kwh
                for info in self.carbon_aware_clients.values()
            ]
        )
        avg_renewable_percentage = np.mean(
            [
                info.renewable_energy_percentage
                for info in self.carbon_aware_clients.values()
            ]
        )

        sustainability_ratings = {}
        for info in self.carbon_aware_clients.values():
            rating = info.sustainability_rating
            sustainability_ratings[rating] = (
                sustainability_ratings.get(rating, 0) + 1
            )

        return {
            "total_carbon_aware_clients": total_clients,
            "average_carbon_intensity": avg_carbon_intensity,
            "average_renewable_percentage": avg_renewable_percentage,
            "sustainability_ratings": sustainability_ratings,
            "clients_within_budget": sum(
                1
                for info in self.carbon_aware_clients.values()
                if info.is_carbon_budget_available()
            ),
        }


class CarbonAwareFederatedServer(FederatedServer):
    """Enhanced federated server with carbon awareness."""

    def __init__(self, config: Optional[CarbonAwareFederatedConfig] = None):
        # Initialize with carbon-aware config
        self.carbon_config = config or CarbonAwareFederatedConfig()
        super().__init__(self.carbon_config)

        # Replace client selector with carbon-aware version
        self.client_selector = CarbonAwareClientSelector(self.carbon_config)

        # Carbon tracking components
        self.carbon_calculator = CarbonCalculator()
        self.energy_tracker = EnergyTracker()
        self.carbon_optimizer = CarbonAwareOptimizer()

        # Carbon monitoring
        self.carbon_reports: List[Dict[str, Any]] = []
        self.total_carbon_consumed = 0.0

        # Start carbon monitoring if enabled
        if self.carbon_config.enable_carbon_monitoring:
            asyncio.create_task(self._carbon_monitoring_loop())

        logger.info("Carbon-aware federated server initialized")

    async def _carbon_monitoring_loop(self):
        """Monitor carbon consumption and generate reports."""
        while True:
            try:
                await asyncio.sleep(
                    self.carbon_config.carbon_reporting_interval
                )

                # Generate carbon report
                carbon_report = self._generate_carbon_report()
                self.carbon_reports.append(carbon_report)

                # Save report if enabled
                if self.carbon_config.save_carbon_reports:
                    await self._save_carbon_report(carbon_report)

                logger.info(
                    f"Generated carbon report: {carbon_report['total_carbon_consumed']:.4f} kg CO2e"
                )

            except Exception as e:
                logger.error(f"Error in carbon monitoring loop: {e}")

    def _generate_carbon_report(self) -> Dict[str, Any]:
        """Generate carbon consumption report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_carbon_consumed": self.total_carbon_consumed,
            "carbon_aware_clients": len(
                self.client_selector.carbon_aware_clients
            ),
            "carbon_summary": self.client_selector.get_carbon_summary(),
            "rounds_completed": len(self.round_history),
            "average_carbon_per_round": self.total_carbon_consumed
            / max(1, len(self.round_history)),
        }

    async def _save_carbon_report(self, report: Dict[str, Any]):
        """Save carbon report to file."""
        try:
            report_dir = Path("sustainability_reports")
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (
                report_dir / f"carbon_aware_federated_report_{timestamp}.json"
            )

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.debug(f"Saved carbon report to {report_file}")

        except Exception as e:
            logger.error(f"Failed to save carbon report: {e}")

    def register_carbon_aware_client(
        self,
        client_id: str,
        public_key: str,
        ip_address: str,
        port: int,
        region: str,
        carbon_budget_limit: float = 1.0,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a client with carbon awareness.

        Args:
            client_id: Unique client identifier
            public_key: Client's public key for authentication
            ip_address: Client's IP address
            port: Client's port number
            region: Client's region for carbon intensity lookup
            carbon_budget_limit: Daily carbon budget limit in kg CO2e
            capabilities: Optional client capabilities

        Returns:
            True if registration successful, False otherwise
        """
        # Register with standard federated server
        success = self.register_client(
            client_id, public_key, ip_address, port, capabilities
        )

        if success:
            # Register with carbon-aware selector
            carbon_success = self.client_selector.register_carbon_aware_client(
                client_id, region, carbon_budget_limit
            )

            if carbon_success:
                logger.info(
                    f"Successfully registered carbon-aware client {client_id} in region {region}"
                )
                return True
            else:
                logger.warning(
                    f"Failed to register carbon awareness for client {client_id}"
                )
                # Still return True as basic registration succeeded

        return success

    def complete_federated_round(
        self, client_updates: List[ModelUpdate]
    ) -> bool:
        """
        Complete federated round with carbon tracking.

        Args:
            client_updates: List of model updates from clients

        Returns:
            True if round completed successfully, False otherwise
        """
        # Calculate carbon consumption for this round
        round_carbon = 0.0
        for update in client_updates:
            # Estimate carbon from energy consumption
            if update.energy_consumed > 0:
                # Get client's carbon intensity
                carbon_info = (
                    self.client_selector.get_carbon_aware_client_info(
                        update.client_id
                    )
                )
                if carbon_info:
                    carbon_intensity = carbon_info.carbon_intensity_gco2_kwh
                else:
                    carbon_intensity = 300.0  # Default

                # Convert energy to carbon
                carbon_kg = (
                    update.energy_consumed * carbon_intensity
                ) / 1000.0  # Convert g to kg
                round_carbon += carbon_kg

                # Update client's carbon budget
                self.client_selector.update_client_carbon_budget(
                    update.client_id, carbon_kg
                )

        # Update total carbon consumption
        self.total_carbon_consumed += round_carbon

        # Complete the round using parent method
        success = super().complete_federated_round(client_updates)

        if success:
            # Log carbon consumption
            logger.info(
                f"Round completed with {round_carbon:.4f} kg CO2e consumed"
            )

            # Check if carbon offset is needed
            if (
                self.carbon_config.enable_automatic_carbon_offset
                and round_carbon > self.carbon_config.carbon_offset_threshold
            ):
                asyncio.create_task(self._trigger_carbon_offset(round_carbon))

        return success

    async def _trigger_carbon_offset(self, carbon_amount: float):
        """Trigger automatic carbon offset for the round."""
        try:
            # This would integrate with the carbon offset marketplace
            logger.info(
                f"Triggering carbon offset for {carbon_amount:.4f} kg CO2e"
            )

            # For demo purposes, just log the offset
            # In production, this would call the carbon offset marketplace
            offset_result = {
                "carbon_amount": carbon_amount,
                "offset_provider": self.carbon_config.carbon_offset_provider,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
            }

            logger.info(f"Carbon offset completed: {offset_result}")

        except Exception as e:
            logger.error(f"Failed to trigger carbon offset: {e}")

    def get_carbon_aware_status(self) -> Dict[str, Any]:
        """Get comprehensive carbon-aware server status."""
        base_status = self.get_server_status()

        carbon_status = {
            "carbon_awareness": {
                "enabled": self.carbon_config.enable_carbon_aware_selection,
                "selection_strategy": self.carbon_config.carbon_selection_strategy.value,
                "total_carbon_consumed": self.total_carbon_consumed,
                "carbon_budget_per_round": self.carbon_config.carbon_budget_per_round,
                "max_carbon_intensity_threshold": self.carbon_config.max_carbon_intensity_threshold,
            },
            "carbon_aware_clients": self.client_selector.get_carbon_summary(),
            "carbon_reports": len(self.carbon_reports),
            "latest_carbon_report": (
                self.carbon_reports[-1] if self.carbon_reports else None
            ),
        }

        # Merge with base status
        base_status.update(carbon_status)
        return base_status


# Utility functions
def create_carbon_aware_federated_server(
    config_dict: Optional[Dict[str, Any]] = None,
) -> CarbonAwareFederatedServer:
    """
    Create a carbon-aware federated server.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configured CarbonAwareFederatedServer instance
    """
    if config_dict:
        config = CarbonAwareFederatedConfig(**config_dict)
    else:
        config = CarbonAwareFederatedConfig()

    return CarbonAwareFederatedServer(config)


async def simulate_carbon_aware_federated_learning(
    num_clients: int = 5,
    num_rounds: int = 10,
    regions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Simulate carbon-aware federated learning.

    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds
        regions: List of regions for clients

    Returns:
        Simulation results
    """
    if regions is None:
        regions = ["US-CA", "US-TX", "EU-DE", "EU-FR", "ASIA-CN"]

    # Create carbon-aware server
    config = CarbonAwareFederatedConfig(
        enable_carbon_aware_selection=True,
        carbon_selection_strategy=CarbonAwareSelectionStrategy.HYBRID,
        max_carbon_intensity_threshold=400.0,
        carbon_budget_per_round=0.1,
    )

    server = create_carbon_aware_federated_server(config.__dict__)

    # Register carbon-aware clients
    for i in range(num_clients):
        client_id = f"carbon_client_{i+1}"
        region = regions[i % len(regions)]

        success = server.register_carbon_aware_client(
            client_id=client_id,
            public_key=f"public_key_{i+1}",
            ip_address="127.0.0.1",
            port=8000 + i,
            region=region,
            carbon_budget_limit=1.0,
        )

        if success:
            logger.info(
                f"Registered carbon-aware client {client_id} in region {region}"
            )

    # Simulate federated rounds
    simulation_results = {
        "carbon_aware_server": server.get_carbon_aware_status(),
        "rounds_completed": 0,
        "total_carbon_consumed": 0.0,
        "carbon_efficiency_improvements": [],
    }

    for round_num in range(num_rounds):
        # Start round
        round_info = server.start_federated_round()
        if round_info is None:
            break

        # Simulate client updates (simplified)
        client_updates = []
        for client_id in round_info.participating_clients:
            # Create mock model update
            mock_update = ModelUpdate(
                client_id=client_id,
                round_number=round_num,
                model_weights={},  # Simplified for demo
                data_size=100,
                training_loss=0.5,
                validation_metrics={"accuracy": 0.85},
                training_time=30.0,
                energy_consumed=0.05,  # 50 Wh
            )
            client_updates.append(mock_update)

        # Complete round
        success = server.complete_federated_round(client_updates)
        if success:
            simulation_results["rounds_completed"] += 1
            simulation_results["total_carbon_consumed"] = (
                server.total_carbon_consumed
            )

    return simulation_results


def compare_carbon_aware_strategies(
    num_clients: int = 5, num_rounds: int = 10
) -> Dict[str, Any]:
    """
    Compare different carbon-aware selection strategies.

    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds

    Returns:
        Comparison results
    """
    strategies = [
        CarbonAwareSelectionStrategy.LOWEST_CARBON,
        CarbonAwareSelectionStrategy.CARBON_BUDGET,
        CarbonAwareSelectionStrategy.HYBRID,
    ]

    results = {}

    for strategy in strategies:
        logger.info(f"Testing carbon-aware strategy: {strategy.value}")

        # Create configuration
        config = CarbonAwareFederatedConfig(
            carbon_selection_strategy=strategy,
            enable_carbon_aware_selection=True,
        )

        # Run simulation
        server = create_carbon_aware_federated_server(config.__dict__)

        # Register clients
        regions = ["US-CA", "US-TX", "EU-DE", "EU-FR", "ASIA-CN"]
        for i in range(num_clients):
            client_id = f"client_{i+1}"
            region = regions[i % len(regions)]

            server.register_carbon_aware_client(
                client_id=client_id,
                public_key=f"public_key_{i+1}",
                ip_address="127.0.0.1",
                port=8000 + i,
                region=region,
            )

        # Simulate rounds
        total_carbon = 0.0
        for round_num in range(num_rounds):
            round_info = server.start_federated_round()
            if round_info:
                # Mock updates
                client_updates = []
                for client_id in round_info.participating_clients:
                    mock_update = ModelUpdate(
                        client_id=client_id,
                        round_number=round_num,
                        model_weights={},
                        data_size=100,
                        training_loss=0.5,
                        validation_metrics={"accuracy": 0.85},
                        training_time=30.0,
                        energy_consumed=0.05,
                    )
                    client_updates.append(mock_update)

                server.complete_federated_round(client_updates)
                total_carbon = server.total_carbon_consumed

        results[strategy.value] = {
            "total_carbon_consumed": total_carbon,
            "carbon_efficiency": 1.0
            / (total_carbon + 0.001),  # Higher is better
            "strategy": strategy.value,
        }

    return results
