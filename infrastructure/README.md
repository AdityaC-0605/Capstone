# 🏗️ Infrastructure

This directory contains all infrastructure-related configuration and deployment files.

## 📁 Directory Structure

```
infrastructure/
├── docker/                  # Docker configuration
│   ├── Dockerfile          # Main application container
│   └── docker-compose.yml  # Multi-service orchestration
├── k8s/                     # Kubernetes manifests
│   ├── api-deployment.yaml # API service deployment
│   ├── postgres.yaml       # Database deployment
│   ├── redis.yaml          # Cache deployment
│   └── monitoring.yaml     # Monitoring stack
├── monitoring/              # Monitoring configuration
│   ├── prometheus.yml      # Metrics collection
│   ├── alert_rules.yml     # Alerting rules
│   └── grafana/            # Dashboard configurations
├── nginx.conf               # Load balancer configuration
└── README.md               # This file
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the application image
docker build -f docker/Dockerfile -t sustainable-ai .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up -d
```

### Services
- **API Service**: Main application server
- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage
- **MLflow**: Model tracking and registry

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (v1.20+)
- kubectl configured
- Helm (optional, for advanced deployments)

### Deploy
```bash
# Deploy all services
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n sustainable-ai
```

### Services
- **API Deployment**: Scalable API service
- **Database**: PostgreSQL with persistent storage
- **Cache**: Redis cluster
- **Monitoring**: Prometheus + Grafana stack

## 📊 Monitoring

### Prometheus
- Collects application metrics
- Stores time-series data
- Provides query interface

### Grafana
- Visualization dashboards
- Alert management
- Performance monitoring

### Alerting
- CPU/Memory usage alerts
- Application error alerts
- Performance threshold alerts

## 🔧 Configuration

### Environment Variables
- `ENVIRONMENT`: development/production
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string
- `LOG_LEVEL`: Logging verbosity

### Secrets Management
- Database credentials
- API keys
- SSL certificates
- JWT secrets

## 🚀 Production Deployment

### Prerequisites
- Production Kubernetes cluster
- SSL certificates
- Domain configuration
- Monitoring setup

### Steps
1. Configure production secrets
2. Deploy infrastructure components
3. Deploy application services
4. Configure monitoring and alerting
5. Run health checks

## 🔒 Security

### Network Security
- TLS/SSL encryption
- Network policies
- Service mesh (optional)

### Access Control
- RBAC configuration
- Service accounts
- Secret management

## 📈 Scaling

### Horizontal Scaling
- API service auto-scaling
- Database read replicas
- Cache clustering

### Vertical Scaling
- Resource limits and requests
- Performance tuning
- Capacity planning

## 🛠️ Maintenance

### Updates
- Rolling deployments
- Database migrations
- Configuration updates

### Backup
- Database backups
- Configuration backups
- Disaster recovery procedures
