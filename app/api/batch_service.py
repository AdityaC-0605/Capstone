"""
Batch Prediction Service for Credit Risk Assessment.

This module implements efficient batch processing capabilities with
asynchronous processing, job queuing, status tracking, and result callbacks.
"""

import json
import time
import uuid
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import warnings

# Queue and async dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Install with: pip install redis")

try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    warnings.warn("Celery not available. Install with: pip install celery")

try:
    from ..core.logging import get_logger, get_audit_logger
    from .inference_service import CreditApplication, PredictionResponse, InferenceService
    from ..sustainability.sustainability_monitor import SustainabilityMonitor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.logging import get_logger, get_audit_logger
    
    # Mock classes for testing
    class MockCreditApplication:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockPredictionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def dict(self):
            return self.__dict__
    
    class MockInferenceService:
        async def _make_single_prediction_internal(self, request):
            return MockPredictionResponse(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                risk_score=0.5,
                confidence=0.8,
                risk_level="medium",
                model_version="1.0.0",
                prediction_timestamp=datetime.now(),
                processing_time_ms=50,
                status="success",
                message="Prediction completed"
            )
    
    CreditApplication = MockCreditApplication
    PredictionResponse = MockPredictionResponse
    InferenceService = MockInferenceService

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class BatchJobStatus(Enum):
    """Batch job status types."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobPriority(Enum):
    """Batch job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchJobConfig:
    """Configuration for batch job processing."""
    max_batch_size: int = 1000
    max_concurrent_jobs: int = 5
    job_timeout_minutes: int = 30
    retry_attempts: int = 3
    chunk_size: int = 100  # Process in chunks
    enable_callbacks: bool = True
    enable_progress_tracking: bool = True
    
    # Queue settings
    queue_backend: str = "memory"  # "memory", "redis", "celery"
    redis_url: Optional[str] = None
    celery_broker: Optional[str] = None
    
    # Storage settings
    result_storage_dir: str = "batch_results"
    keep_results_days: int = 7


@dataclass
class BatchJob:
    """Batch job container."""
    job_id: str
    applications: List[Dict[str, Any]]
    status: BatchJobStatus = BatchJobStatus.PENDING
    priority: BatchJobPriority = BatchJobPriority.NORMAL
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Configuration
    include_explanations: bool = False
    explanation_type: str = "shap"
    track_sustainability: bool = True
    
    # Progress tracking
    total_applications: int = 0
    processed_applications: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    # Results
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Callbacks
    callback_url: Optional[str] = None
    callback_headers: Dict[str, str] = field(default_factory=dict)
    
    # Sustainability
    sustainability_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.total_applications == 0:
            self.total_applications = len(self.applications)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_applications == 0:
            return 0.0
        return (self.processed_applications / self.total_applications) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed_applications == 0:
            return 0.0
        return (self.successful_predictions / self.processed_applications) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_applications": self.total_applications,
            "processed_applications": self.processed_applications,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "progress_percentage": self.progress_percentage,
            "success_rate": self.success_rate,
            "include_explanations": self.include_explanations,
            "explanation_type": self.explanation_type,
            "track_sustainability": self.track_sustainability,
            "callback_url": self.callback_url,
            "sustainability_metrics": self.sustainability_metrics,
            "errors": self.errors[-10:]  # Last 10 errors only
        }


class BatchQueue:
    """Abstract base class for batch job queues."""
    
    def __init__(self, config: BatchJobConfig):
        self.config = config
    
    async def enqueue(self, job: BatchJob) -> str:
        """Enqueue a batch job."""
        raise NotImplementedError
    
    async def dequeue(self) -> Optional[BatchJob]:
        """Dequeue the next batch job."""
        raise NotImplementedError
    
    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        raise NotImplementedError
    
    async def update_job(self, job: BatchJob) -> None:
        """Update job status and progress."""
        raise NotImplementedError
    
    async def list_jobs(self, status: Optional[BatchJobStatus] = None) -> List[BatchJob]:
        """List jobs, optionally filtered by status."""
        raise NotImplementedError


class MemoryBatchQueue(BatchQueue):
    """In-memory batch job queue implementation."""
    
    def __init__(self, config: BatchJobConfig):
        super().__init__(config)
        self.jobs: Dict[str, BatchJob] = {}
        self.queue: List[str] = []  # Job IDs in priority order
        self.lock = asyncio.Lock()
    
    async def enqueue(self, job: BatchJob) -> str:
        """Enqueue a batch job."""
        async with self.lock:
            self.jobs[job.job_id] = job
            job.status = BatchJobStatus.QUEUED
            
            # Insert based on priority
            inserted = False
            for i, existing_job_id in enumerate(self.queue):
                existing_job = self.jobs[existing_job_id]
                if job.priority.value > existing_job.priority.value:
                    self.queue.insert(i, job.job_id)
                    inserted = True
                    break
            
            if not inserted:
                self.queue.append(job.job_id)
            
            logger.info(f"Job {job.job_id} enqueued with priority {job.priority.value}")
            return job.job_id
    
    async def dequeue(self) -> Optional[BatchJob]:
        """Dequeue the next batch job."""
        async with self.lock:
            if not self.queue:
                return None
            
            job_id = self.queue.pop(0)
            job = self.jobs.get(job_id)
            
            if job and job.status == BatchJobStatus.QUEUED:
                job.status = BatchJobStatus.PROCESSING
                job.started_at = datetime.now()
                return job
            
            return None
    
    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    async def update_job(self, job: BatchJob) -> None:
        """Update job status and progress."""
        async with self.lock:
            self.jobs[job.job_id] = job
    
    async def list_jobs(self, status: Optional[BatchJobStatus] = None) -> List[BatchJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status:
            jobs = [job for job in jobs if job.status == status]
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)


class RedisBatchQueue(BatchQueue):
    """Redis-based batch job queue implementation."""
    
    def __init__(self, config: BatchJobConfig):
        super().__init__(config)
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for RedisBatchQueue")
        
        self.redis_client = redis.from_url(config.redis_url or "redis://localhost:6379")
        self.job_key_prefix = "batch_job:"
        self.queue_key = "batch_queue"
    
    async def enqueue(self, job: BatchJob) -> str:
        """Enqueue a batch job."""
        job.status = BatchJobStatus.QUEUED
        
        # Store job data
        job_key = f"{self.job_key_prefix}{job.job_id}"
        job_data = json.dumps(job.to_dict())
        self.redis_client.set(job_key, job_data)
        
        # Add to priority queue
        priority_score = job.priority.value * 1000 + int(time.time())
        self.redis_client.zadd(self.queue_key, {job.job_id: priority_score})
        
        logger.info(f"Job {job.job_id} enqueued to Redis")
        return job.job_id
    
    async def dequeue(self) -> Optional[BatchJob]:
        """Dequeue the next batch job."""
        # Get highest priority job
        result = self.redis_client.zpopmax(self.queue_key)
        if not result:
            return None
        
        job_id, _ = result[0]
        job_key = f"{self.job_key_prefix}{job_id}"
        job_data = self.redis_client.get(job_key)
        
        if not job_data:
            return None
        
        # Reconstruct job object
        job_dict = json.loads(job_data)
        job = self._dict_to_job(job_dict)
        job.status = BatchJobStatus.PROCESSING
        job.started_at = datetime.now()
        
        # Update in Redis
        await self.update_job(job)
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        job_key = f"{self.job_key_prefix}{job_id}"
        job_data = self.redis_client.get(job_key)
        
        if not job_data:
            return None
        
        job_dict = json.loads(job_data)
        return self._dict_to_job(job_dict)
    
    async def update_job(self, job: BatchJob) -> None:
        """Update job status and progress."""
        job_key = f"{self.job_key_prefix}{job.job_id}"
        job_data = json.dumps(job.to_dict())
        self.redis_client.set(job_key, job_data)
    
    async def list_jobs(self, status: Optional[BatchJobStatus] = None) -> List[BatchJob]:
        """List jobs, optionally filtered by status."""
        # Get all job keys
        job_keys = self.redis_client.keys(f"{self.job_key_prefix}*")
        jobs = []
        
        for job_key in job_keys:
            job_data = self.redis_client.get(job_key)
            if job_data:
                job_dict = json.loads(job_data)
                job = self._dict_to_job(job_dict)
                
                if not status or job.status == status:
                    jobs.append(job)
        
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    def _dict_to_job(self, job_dict: Dict[str, Any]) -> BatchJob:
        """Convert dictionary back to BatchJob object."""
        # This is a simplified conversion - in practice, you'd need more robust deserialization
        job = BatchJob(
            job_id=job_dict["job_id"],
            applications=[],  # Applications would be stored separately for large batches
            status=BatchJobStatus(job_dict["status"]),
            priority=BatchJobPriority(job_dict["priority"]),
            created_at=datetime.fromisoformat(job_dict["created_at"]),
            include_explanations=job_dict["include_explanations"],
            explanation_type=job_dict["explanation_type"],
            track_sustainability=job_dict["track_sustainability"]
        )
        
        if job_dict["started_at"]:
            job.started_at = datetime.fromisoformat(job_dict["started_at"])
        if job_dict["completed_at"]:
            job.completed_at = datetime.fromisoformat(job_dict["completed_at"])
        
        job.total_applications = job_dict["total_applications"]
        job.processed_applications = job_dict["processed_applications"]
        job.successful_predictions = job_dict["successful_predictions"]
        job.failed_predictions = job_dict["failed_predictions"]
        job.callback_url = job_dict.get("callback_url")
        job.sustainability_metrics = job_dict.get("sustainability_metrics")
        job.errors = job_dict.get("errors", [])
        
        return job


class BatchProcessor:
    """Batch job processor with async processing capabilities."""
    
    def __init__(self, config: BatchJobConfig, inference_service: Optional[Any] = None):
        self.config = config
        if inference_service:
            self.inference_service = inference_service
        else:
            # Create mock classes for testing
            class MockPredictionResponse:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                
                def dict(self):
                    return self.__dict__
            
            class MockInferenceService:
                async def _make_single_prediction_internal(self, request):
                    return MockPredictionResponse(
                        prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                        risk_score=0.5,
                        confidence=0.8,
                        risk_level="medium",
                        model_version="1.0.0",
                        prediction_timestamp=datetime.now(),
                        processing_time_ms=50,
                        status="success",
                        message="Prediction completed"
                    )
            self.inference_service = MockInferenceService()
        try:
            self.sustainability_monitor = SustainabilityMonitor()
        except:
            self.sustainability_monitor = None
        
        # Initialize queue
        if config.queue_backend == "redis" and REDIS_AVAILABLE:
            self.queue = RedisBatchQueue(config)
        else:
            self.queue = MemoryBatchQueue(config)
        
        # Processing state
        self.is_processing = False
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.processing_lock = asyncio.Lock()
        
        # Create results directory
        Path(config.result_storage_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch processor initialized with {config.queue_backend} queue")
    
    async def submit_batch_job(self, applications: List[Dict[str, Any]], 
                             priority: BatchJobPriority = BatchJobPriority.NORMAL,
                             include_explanations: bool = False,
                             explanation_type: str = "shap",
                             track_sustainability: bool = True,
                             callback_url: Optional[str] = None,
                             callback_headers: Optional[Dict[str, str]] = None) -> str:
        """Submit a new batch job."""
        
        # Validate batch size
        if len(applications) > self.config.max_batch_size:
            raise ValueError(f"Batch size {len(applications)} exceeds maximum {self.config.max_batch_size}")
        
        # Create job
        job_id = f"batch_{uuid.uuid4().hex}"
        job = BatchJob(
            job_id=job_id,
            applications=applications,
            priority=priority,
            include_explanations=include_explanations,
            explanation_type=explanation_type,
            track_sustainability=track_sustainability,
            callback_url=callback_url,
            callback_headers=callback_headers or {}
        )
        
        # Enqueue job
        await self.queue.enqueue(job)
        
        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self.start_processing())
        
        # Log job submission
        audit_logger.log_model_operation(
            user_id="batch_processor",
            model_id="batch_service",
            operation="submit_batch_job",
            success=True,
            details={
                "job_id": job_id,
                "batch_size": len(applications),
                "priority": priority.value,
                "include_explanations": include_explanations
            }
        )
        
        logger.info(f"Batch job {job_id} submitted with {len(applications)} applications")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and progress."""
        job = await self.queue.get_job(job_id)
        if not job:
            return None
        
        return job.to_dict()
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = await self.queue.get_job(job_id)
        if not job:
            return False
        
        if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]:
            return False
        
        # Cancel active processing task
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            del self.active_jobs[job_id]
        
        # Update job status
        job.status = BatchJobStatus.CANCELLED
        job.completed_at = datetime.now()
        await self.queue.update_job(job)
        
        logger.info(f"Batch job {job_id} cancelled")
        return True
    
    async def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List batch jobs."""
        status_enum = BatchJobStatus(status) if status else None
        jobs = await self.queue.list_jobs(status_enum)
        return [job.to_dict() for job in jobs]
    
    async def start_processing(self):
        """Start the batch processing loop."""
        if self.is_processing:
            return
        
        async with self.processing_lock:
            self.is_processing = True
            logger.info("Batch processing started")
            
            try:
                while self.is_processing:
                    # Check if we can process more jobs
                    if len(self.active_jobs) >= self.config.max_concurrent_jobs:
                        await asyncio.sleep(1)
                        continue
                    
                    # Get next job
                    job = await self.queue.dequeue()
                    if not job:
                        await asyncio.sleep(5)  # Wait before checking again
                        continue
                    
                    # Start processing job
                    task = asyncio.create_task(self._process_job(job))
                    self.active_jobs[job.job_id] = task
                    
                    # Clean up completed tasks
                    completed_jobs = [job_id for job_id, task in self.active_jobs.items() if task.done()]
                    for job_id in completed_jobs:
                        del self.active_jobs[job_id]
                    
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
            finally:
                self.is_processing = False
                logger.info("Batch processing stopped")
    
    async def stop_processing(self):
        """Stop batch processing."""
        self.is_processing = False
        
        # Cancel all active jobs
        for task in self.active_jobs.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
        
        self.active_jobs.clear()
        logger.info("Batch processing stopped and all jobs cancelled")
    
    async def _process_job(self, job: BatchJob):
        """Process a single batch job."""
        logger.info(f"Processing batch job {job.job_id} with {job.total_applications} applications")
        
        try:
            # Start sustainability tracking if enabled
            sustainability_context = None
            if job.track_sustainability and self.sustainability_monitor:
                sustainability_context = self.sustainability_monitor.start_experiment_tracking(
                    f"batch_{job.job_id}",
                    {"type": "batch_prediction", "batch_size": job.total_applications}
                )
            
            # Process applications in chunks
            chunk_size = self.config.chunk_size
            for i in range(0, len(job.applications), chunk_size):
                chunk = job.applications[i:i + chunk_size]
                await self._process_chunk(job, chunk)
                
                # Update progress
                await self.queue.update_job(job)
                
                # Check for cancellation
                if job.status == BatchJobStatus.CANCELLED:
                    break
            
            # Complete job
            if job.status != BatchJobStatus.CANCELLED:
                job.status = BatchJobStatus.COMPLETED
                job.completed_at = datetime.now()
            
            # Stop sustainability tracking
            if sustainability_context and self.sustainability_monitor:
                job.sustainability_metrics = self.sustainability_monitor.stop_experiment_tracking(sustainability_context)
            
            # Save results
            await self._save_job_results(job)
            
            # Send callback if configured
            if job.callback_url:
                await self._send_callback(job)
            
            # Final update
            await self.queue.update_job(job)
            
            logger.info(f"Batch job {job.job_id} completed: {job.successful_predictions}/{job.total_applications} successful")
            
        except Exception as e:
            logger.error(f"Error processing batch job {job.job_id}: {e}")
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.now()
            job.errors.append(f"Job processing failed: {str(e)}")
            await self.queue.update_job(job)
    
    async def _process_chunk(self, job: BatchJob, chunk: List[Dict[str, Any]]):
        """Process a chunk of applications."""
        
        for app_data in chunk:
            try:
                # Create prediction request
                from .inference_service import PredictionRequest, CreditApplication
                
                # Convert dict to CreditApplication
                credit_app = CreditApplication(**app_data)
                
                prediction_request = PredictionRequest(
                    application=credit_app,
                    include_explanation=job.include_explanations,
                    explanation_type=job.explanation_type,
                    track_sustainability=False  # Already tracking at job level
                )
                
                # Make prediction
                prediction = await self.inference_service._make_single_prediction_internal(prediction_request)
                
                # Store result
                job.predictions.append(prediction.dict() if hasattr(prediction, 'dict') else prediction.__dict__)
                job.successful_predictions += 1
                
            except Exception as e:
                logger.error(f"Error processing application in job {job.job_id}: {e}")
                job.errors.append(f"Application processing failed: {str(e)}")
                job.failed_predictions += 1
            
            job.processed_applications += 1
    
    async def _save_job_results(self, job: BatchJob):
        """Save job results to file."""
        try:
            results_file = Path(self.config.result_storage_dir) / f"{job.job_id}_results.json"
            
            results_data = {
                "job_info": job.to_dict(),
                "predictions": job.predictions,
                "summary": {
                    "total_applications": job.total_applications,
                    "successful_predictions": job.successful_predictions,
                    "failed_predictions": job.failed_predictions,
                    "success_rate": job.success_rate,
                    "processing_time_seconds": (job.completed_at - job.started_at).total_seconds() if job.completed_at and job.started_at else 0
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Results saved for job {job.job_id}: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results for job {job.job_id}: {e}")
    
    async def _send_callback(self, job: BatchJob):
        """Send callback notification."""
        try:
            import aiohttp
            
            callback_data = {
                "job_id": job.job_id,
                "status": job.status.value,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "summary": {
                    "total_applications": job.total_applications,
                    "successful_predictions": job.successful_predictions,
                    "failed_predictions": job.failed_predictions,
                    "success_rate": job.success_rate
                },
                "sustainability_metrics": job.sustainability_metrics
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    job.callback_url,
                    json=callback_data,
                    headers=job.callback_headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Callback sent successfully for job {job.job_id}")
                    else:
                        logger.warning(f"Callback failed for job {job.job_id}: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending callback for job {job.job_id}: {e}")


# Utility functions

def create_batch_processor(config: Optional[BatchJobConfig] = None,
                          inference_service: Optional[Any] = None) -> BatchProcessor:
    """Create batch processor instance."""
    config = config or BatchJobConfig()
    return BatchProcessor(config, inference_service)


async def submit_batch_job(applications: List[Dict[str, Any]],
                          processor: Optional[BatchProcessor] = None,
                          **kwargs) -> str:
    """Submit a batch job for processing."""
    if processor is None:
        processor = create_batch_processor()
    
    return await processor.submit_batch_job(applications, **kwargs)