# Kubernetes Deployment for Sustainable Credit Risk AI System

This directory contains Kubernetes manifests and deployment scripts for the Sustainable Credit Risk AI System.

## Architecture Overview

The system is deployed as a microservices architecture on Kubernetes with the following components:

### Core Services
- **API Service**: FastAPI-based inference service with horizontal pod autoscaling
- **Federated Learning Server**: Coordinates federated training across multiple clients
- **Training Jobs**: Batch and scheduled training jobs with GPU support

### Infrastructure Services
- **PostgreSQL**: Primary database for application data and MLflow backend
- **Redis**: Caching and session storage
- **MLflow**: ML experiment tracking and model registry

### Monitoring Stack
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization dashboards and alerting

## Prerequisites

1. **Kubernetes Cluster**: Version 1.20+ with the following features:
   - Horizontal Pod Autoscaler (HPA)
   - Persistent Volume support
   - Ingress controller (NGINX recommended)
   - GPU support (for training workloads)

2. **Required Tools**:
   ```bash
   kubectl >= 1.20
   kustomize >= 4.0 (optional)
   helm >= 3.0 (for monitoring stack)
   ```

3. **Container Images**: Build and push the following images to your registry:
   ```bash
   sustainable-credit-risk-ai:production
   sustainable-credit-risk-ai:development
   sustainable-credit-risk-ai:training
   ```

## Quick Start

### 1. Deploy the Complete System

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Deploy all components
./deploy.sh deploy
```

### 2. Check Deployment Status

```bash
./deploy.sh status
```

### 3. Run Training Job

```bash
./deploy.sh training
```

### 4. Clean Up

```bash
./deploy.sh clean
```

## Manual Deployment

If you prefer manual deployment, follow these steps:

### 1. Create Namespace and Basic Resources

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f rbac.yaml
kubectl apply -f persistent-volumes.yaml
```

### 2. Deploy Infrastructure Services

```bash
# Deploy PostgreSQL
kubectl apply -f postgres-deployment.yaml
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s

# Deploy Redis
kubectl apply -f redis-deployment.yaml
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s

# Deploy MLflow
kubectl apply -f mlflow-deployment.yaml
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s
```

### 3. Deploy Application Services

```bash
# Deploy API service
kubectl apply -f api-deployment.yaml
kubectl wait --for=condition=ready pod -l app=api --timeout=300s

# Deploy Federated Learning Server
kubectl apply -f federated-server-deployment.yaml
kubectl wait --for=condition=ready pod -l app=federated-server --timeout=300s
```

### 4. Deploy Monitoring

```bash
kubectl apply -f monitoring-deployment.yaml
kubectl wait --for=condition=ready pod -l app=prometheus --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana --timeout=300s
```

### 5. Setup Ingress

```bash
kubectl apply -f ingress.yaml
```

## Configuration

### Environment Variables

Key configuration is managed through ConfigMaps and Secrets:

- **ConfigMap `credit-risk-config`**: Application configuration
- **Secret `credit-risk-secrets`**: Sensitive data (passwords, keys)
- **ConfigMap `postgres-config`**: Database configuration

### Persistent Storage

The system uses persistent volumes for:
- PostgreSQL data (`postgres-pvc`: 10Gi)
- Redis data (`redis-pvc`: 5Gi)
- MLflow artifacts (`mlflow-artifacts-pvc`: 20Gi)
- Model storage (`model-storage-pvc`: 50Gi)

### Resource Requirements

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| API | 500m | 1Gi | 1000m | 2Gi |
| Federated Server | 1000m | 2Gi | 2000m | 4Gi |
| Training Job | 2000m | 4Gi | 4000m | 8Gi |
| PostgreSQL | 250m | 512Mi | 500m | 1Gi |
| Redis | 100m | 256Mi | 200m | 512Mi |
| MLflow | 250m | 512Mi | 500m | 1Gi |
| Prometheus | 250m | 512Mi | 500m | 1Gi |
| Grafana | 100m | 256Mi | 200m | 512Mi |

## Scaling

### Horizontal Pod Autoscaler

The API service includes HPA configuration:
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

### Manual Scaling

```bash
# Scale API service
kubectl scale deployment api-deployment --replicas=5

# Scale federated server
kubectl scale deployment federated-server-deployment --replicas=2
```

## Monitoring and Observability

### Accessing Monitoring Services

1. **Grafana Dashboard**:
   ```bash
   kubectl port-forward svc/grafana-service 3000:3000
   # Access: http://localhost:3000 (admin/admin)
   ```

2. **Prometheus**:
   ```bash
   kubectl port-forward svc/prometheus-service 9090:9090
   # Access: http://localhost:9090
   ```

3. **MLflow**:
   ```bash
   kubectl port-forward svc/mlflow-service 5000:5000
   # Access: http://localhost:5000
   ```

### Health Checks

All services include health checks:
- **Liveness Probes**: Restart unhealthy containers
- **Readiness Probes**: Remove unhealthy pods from load balancing

## Security

### Network Policies

The deployment includes network policies to:
- Restrict inter-pod communication
- Allow only necessary traffic flows
- Isolate sensitive components

### RBAC

Role-Based Access Control is configured with:
- Service accounts for each component
- Minimal required permissions
- Cluster-level access for monitoring

### Secrets Management

Sensitive data is stored in Kubernetes secrets:
- Database passwords
- JWT signing keys
- Encryption keys
- TLS certificates

## Service Mesh (Optional)

For advanced traffic management, the system supports Istio service mesh:

```bash
# Install Istio (if not already installed)
istioctl install --set values.defaultRevision=default

# Enable sidecar injection
kubectl label namespace credit-risk-ai istio-injection=enabled

# Apply service mesh configuration
kubectl apply -f service-mesh.yaml
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending state**:
   ```bash
   kubectl describe pod <pod-name>
   # Check for resource constraints or PVC issues
   ```

2. **Service connectivity issues**:
   ```bash
   kubectl get endpoints
   kubectl describe service <service-name>
   ```

3. **Persistent volume issues**:
   ```bash
   kubectl get pv,pvc
   kubectl describe pvc <pvc-name>
   ```

### Logs

```bash
# View application logs
kubectl logs -f deployment/api-deployment

# View training job logs
kubectl logs job/model-training-job

# View all pods logs
kubectl logs -l app=api --tail=100
```

### Debug Mode

For debugging, you can run a debug pod:

```bash
kubectl run debug --image=busybox:1.35 --rm -it --restart=Never -- sh
```

## Backup and Recovery

### Database Backup

```bash
# Create database backup
kubectl exec deployment/postgres-deployment -- pg_dump -U postgres credit_risk_ai > backup.sql

# Restore database
kubectl exec -i deployment/postgres-deployment -- psql -U postgres credit_risk_ai < backup.sql
```

### Model Backup

Models are stored in persistent volumes and can be backed up using volume snapshots or by copying to external storage.

## Production Considerations

1. **Use external databases** for production (managed PostgreSQL, Redis)
2. **Configure proper resource limits** based on workload requirements
3. **Set up monitoring and alerting** for all critical components
4. **Implement proper backup strategies** for data and models
5. **Use secrets management solutions** like HashiCorp Vault
6. **Configure network policies** for security
7. **Set up log aggregation** (ELK stack, Fluentd)
8. **Use managed Kubernetes services** (EKS, GKE, AKS) for production

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Kubernetes events: `kubectl get events --sort-by=.metadata.creationTimestamp`
3. Check application logs for specific error messages
4. Verify resource availability and constraints