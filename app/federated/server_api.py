"""
Federated Learning Server API.

This module implements REST API endpoints for the federated learning server,
providing HTTP interfaces for client registration, model updates, and coordination.
"""

import asyncio
import json
import logging
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

try:
    from ..core.logging import get_audit_logger, get_logger
    from .communication import (
        FederatedCommunicationManager,
        FederatedMessage,
        MessageSerializer,
        MessageType,
        create_registration_message,
    )
    from .federated_server import (
        ClientInfo,
        FederatedConfig,
        FederatedServer,
        ModelUpdate,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger
    from federated.communication import (
        FederatedCommunicationManager,
        FederatedMessage,
        MessageSerializer,
        MessageType,
        create_registration_message,
    )
    from federated.federated_server import (
        ClientInfo,
        FederatedConfig,
        FederatedServer,
        ModelUpdate,
    )

logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Security
security = HTTPBearer(auto_error=False)


# Pydantic models for API requests/responses
class ClientRegistrationRequest(BaseModel):
    """Request model for client registration."""

    client_id: str = Field(..., description="Unique client identifier")
    public_key: str = Field(..., description="Client's public key in PEM format")
    ip_address: str = Field(..., description="Client's IP address")
    port: int = Field(..., description="Client's port number")
    capabilities: Optional[Dict[str, Any]] = Field(
        default={}, description="Client capabilities"
    )
    data_size: Optional[int] = Field(
        default=0, description="Size of client's training data"
    )


class ClientRegistrationResponse(BaseModel):
    """Response model for client registration."""

    success: bool
    message: str
    authentication_token: Optional[str] = None
    server_public_key: Optional[str] = None
    client_id: str


class ModelUpdateRequest(BaseModel):
    """Request model for model updates."""

    client_id: str
    round_number: int
    model_weights_encoded: str = Field(..., description="Base64 encoded model weights")
    data_size: int
    training_loss: float
    validation_metrics: Dict[str, float]
    training_time: float
    energy_consumed: Optional[float] = 0.0
    differential_privacy_applied: Optional[bool] = False
    epsilon_used: Optional[float] = 0.0


class ModelUpdateResponse(BaseModel):
    """Response model for model updates."""

    success: bool
    message: str
    round_number: int
    next_round_ready: bool = False


class GlobalModelResponse(BaseModel):
    """Response model for global model requests."""

    success: bool
    model_weights_encoded: Optional[str] = None
    model_version: int
    round_number: int
    message: str


class ServerStatusResponse(BaseModel):
    """Response model for server status."""

    server_info: Dict[str, Any]
    clients: Dict[str, Any]
    training: Dict[str, Any]
    security: Dict[str, Any]


class FederatedServerAPI:
    """FastAPI application for federated learning server."""

    def __init__(
        self,
        federated_server: FederatedServer,
        config: Optional[FederatedConfig] = None,
    ):
        self.server = federated_server
        self.config = config or FederatedConfig()
        self.communication_manager = FederatedCommunicationManager(self.config)

        # Create FastAPI app
        self.app = FastAPI(
            title="Federated Learning Server API",
            description="REST API for federated learning coordination",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

        # Background tasks
        self.background_tasks = set()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.post("/api/v1/register", response_model=ClientRegistrationResponse)
        async def register_client(
            request: ClientRegistrationRequest,
            background_tasks: BackgroundTasks,
            client_request: Request,
        ):
            """Register a new federated client."""
            try:
                client_ip = client_request.client.host

                # Register client with server
                success = self.server.register_client(
                    client_id=request.client_id,
                    public_key=request.public_key,
                    ip_address=client_ip,
                    port=request.port,
                    capabilities=request.capabilities,
                )

                if success:
                    # Get client info for response
                    client_info = self.server.clients.get(request.client_id)

                    # Add client's public key to communication manager
                    self.communication_manager.communicator.add_peer_public_key(
                        request.client_id, request.public_key
                    )

                    # Update client data size if provided
                    if request.data_size > 0 and client_info:
                        client_info.data_size = request.data_size

                    response = ClientRegistrationResponse(
                        success=True,
                        message="Client registered successfully",
                        authentication_token=(
                            client_info.authentication_token if client_info else None
                        ),
                        server_public_key=self.communication_manager.communicator.get_public_key_pem(),
                        client_id=request.client_id,
                    )

                    logger.info(
                        f"Client {request.client_id} registered from {client_ip}"
                    )
                    return response
                else:
                    raise HTTPException(
                        status_code=400, detail="Failed to register client"
                    )

            except Exception as e:
                logger.error(f"Registration error for client {request.client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/v1/unregister/{client_id}")
        async def unregister_client(
            client_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security),
        ):
            """Unregister a federated client."""
            try:
                # Authenticate client
                if not self._authenticate_client(client_id, credentials):
                    raise HTTPException(status_code=401, detail="Authentication failed")

                success = self.server.unregister_client(client_id)

                if success:
                    return {
                        "success": True,
                        "message": "Client unregistered successfully",
                    }
                else:
                    raise HTTPException(status_code=404, detail="Client not found")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unregistration error for client {client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/global-model", response_model=GlobalModelResponse)
        async def get_global_model(
            client_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security),
        ):
            """Get the current global model."""
            try:
                # Authenticate client
                if not self._authenticate_client(client_id, credentials):
                    raise HTTPException(status_code=401, detail="Authentication failed")

                # Get global model weights
                global_weights = self.server.get_global_model_weights()

                if global_weights:
                    # Serialize weights
                    encoded_weights = MessageSerializer.serialize_model_weights(
                        global_weights
                    )

                    return GlobalModelResponse(
                        success=True,
                        model_weights_encoded=encoded_weights,
                        model_version=self.server.current_round,
                        round_number=self.server.current_round,
                        message="Global model retrieved successfully",
                    )
                else:
                    return GlobalModelResponse(
                        success=False,
                        model_version=0,
                        round_number=0,
                        message="No global model available",
                    )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting global model for client {client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/model-update", response_model=ModelUpdateResponse)
        async def submit_model_update(
            request: ModelUpdateRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(security),
        ):
            """Submit a model update from a client."""
            try:
                # Authenticate client
                if not self._authenticate_client(request.client_id, credentials):
                    raise HTTPException(status_code=401, detail="Authentication failed")

                # Deserialize model weights
                model_weights = MessageSerializer.deserialize_model_weights(
                    request.model_weights_encoded
                )

                # Create model update object
                model_update = ModelUpdate(
                    client_id=request.client_id,
                    round_number=request.round_number,
                    model_weights=model_weights,
                    data_size=request.data_size,
                    training_loss=request.training_loss,
                    validation_metrics=request.validation_metrics,
                    training_time=request.training_time,
                    energy_consumed=request.energy_consumed,
                    differential_privacy_applied=request.differential_privacy_applied,
                    epsilon_used=request.epsilon_used,
                )

                # Calculate and set model hash for integrity
                model_update.model_hash = model_update.calculate_hash()

                # Add to background processing
                background_tasks.add_task(self._process_model_update, model_update)

                return ModelUpdateResponse(
                    success=True,
                    message="Model update received and queued for processing",
                    round_number=request.round_number,
                    next_round_ready=False,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    f"Error processing model update from client {request.client_id}: {e}"
                )
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/status", response_model=ServerStatusResponse)
        async def get_server_status():
            """Get comprehensive server status."""
            try:
                status = self.server.get_server_status()
                return ServerStatusResponse(**status)

            except Exception as e:
                logger.error(f"Error getting server status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/start-round")
        async def start_training_round(
            credentials: HTTPAuthorizationCredentials = Depends(security),
        ):
            """Start a new federated training round."""
            try:
                # This endpoint might be restricted to admin users
                # For now, we'll allow any authenticated request

                round_info = self.server.start_federated_round()

                if round_info:
                    return {
                        "success": True,
                        "message": f"Started round {round_info.round_number}",
                        "round_number": round_info.round_number,
                        "participating_clients": round_info.participating_clients,
                    }
                else:
                    return {"success": False, "message": "Failed to start new round"}

            except Exception as e:
                logger.error(f"Error starting training round: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/round/{round_number}")
        async def get_round_info(round_number: int):
            """Get information about a specific training round."""
            try:
                if round_number < len(self.server.round_history):
                    round_info = self.server.round_history[round_number]

                    return {
                        "round_number": round_info.round_number,
                        "start_time": round_info.start_time.isoformat(),
                        "end_time": (
                            round_info.end_time.isoformat()
                            if round_info.end_time
                            else None
                        ),
                        "participating_clients": round_info.participating_clients,
                        "is_complete": round_info.is_complete(),
                        "duration_seconds": round_info.get_duration(),
                        "total_data_size": round_info.total_data_size,
                        "aggregated_metrics": round_info.aggregated_metrics,
                    }
                else:
                    raise HTTPException(status_code=404, detail="Round not found")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting round info for round {round_number}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/heartbeat")
        async def client_heartbeat(
            client_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security),
        ):
            """Receive heartbeat from client."""
            try:
                # Authenticate client
                if not self._authenticate_client(client_id, credentials):
                    raise HTTPException(status_code=401, detail="Authentication failed")

                # Update client last seen
                if client_id in self.server.clients:
                    self.server.clients[client_id].update_last_seen()
                    return {"success": True, "message": "Heartbeat received"}
                else:
                    raise HTTPException(status_code=404, detail="Client not registered")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing heartbeat from client {client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "server_running": self.server.is_running,
                "current_round": self.server.current_round,
                "registered_clients": len(self.server.clients),
            }

    def _authenticate_client(
        self, client_id: str, credentials: Optional[HTTPAuthorizationCredentials]
    ) -> bool:
        """Authenticate a client request."""
        if not self.config.require_authentication:
            return True

        if not credentials:
            return False

        return self.server.authenticate_client(client_id, credentials.credentials)

    async def _process_model_update(self, model_update: ModelUpdate):
        """Process a model update in the background."""
        try:
            # This is a simplified version - in practice, you might want to:
            # 1. Store the update temporarily
            # 2. Wait for enough updates to arrive
            # 3. Trigger aggregation when ready
            # 4. Notify clients of new global model

            logger.info(
                f"Processing model update from client {model_update.client_id} "
                f"for round {model_update.round_number}"
            )

            # For now, just log the update
            audit_logger.log_model_operation(
                user_id=model_update.client_id,
                model_id="federated_model",
                operation="model_update_received",
                success=True,
                details={
                    "round_number": model_update.round_number,
                    "data_size": model_update.data_size,
                    "training_loss": model_update.training_loss,
                    "training_time": model_update.training_time,
                    "energy_consumed": model_update.energy_consumed,
                },
            )

        except Exception as e:
            logger.error(
                f"Error processing model update from {model_update.client_id}: {e}"
            )

    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs):
        """Run the federated server API."""
        logger.info(f"Starting federated server API on {host}:{port}")

        # Mark server as running
        self.server.is_running = True

        # Run the server
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Utility functions
def create_federated_server_api(
    config_dict: Optional[Dict[str, Any]] = None,
) -> FederatedServerAPI:
    """
    Create a federated server API with configuration.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configured FederatedServerAPI instance
    """
    # Create federated server
    if config_dict:
        config = FederatedConfig(**config_dict)
    else:
        config = FederatedConfig()

    federated_server = FederatedServer(config)

    # Create API
    return FederatedServerAPI(federated_server, config)


async def run_federated_server_async(
    api: FederatedServerAPI, host: str = "0.0.0.0", port: int = 8080
):
    """
    Run federated server API asynchronously.

    Args:
        api: FederatedServerAPI instance
        host: Host to bind to
        port: Port to bind to
    """
    config = uvicorn.Config(api.app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    # Mark server as running
    api.server.is_running = True

    try:
        await server.serve()
    finally:
        api.server.is_running = False


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run Federated Learning Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--max-clients", type=int, default=10, help="Maximum number of clients"
    )
    parser.add_argument(
        "--require-auth", action="store_true", help="Require client authentication"
    )
    parser.add_argument(
        "--enable-encryption", action="store_true", help="Enable message encryption"
    )

    args = parser.parse_args()

    # Create configuration
    config_dict = {
        "server_host": args.host,
        "server_port": args.port,
        "max_clients": args.max_clients,
        "require_authentication": args.require_auth,
        "enable_encryption": args.enable_encryption,
    }

    # Create and run server
    api = create_federated_server_api(config_dict)
    api.run(host=args.host, port=args.port)
