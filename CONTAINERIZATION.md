# Containerization Guide

This document provides comprehensive guidance for containerizing and deploying the Sustainable Credit Risk AI System.

## 🐳 Container Architecture

### Multi-Stage Dockerfile

The system uses a multi-stage Dockerfile with optimized builds for different environments:

- **Base Stage**: Common Python environment with dependencies
- **Development Stage**: Full development environment with dev tools
- **Production Stage**: Optimized production environment
- **Training Stage**: ML training environment with additional tools
- **Inference Stage**: Lightweight inference-only environment

### Container Images

```bash
# Build different stages
docker build --target development -t credit-risk-ai:dev .
docker build --target production -t credit-risk-ai:prod .
docker build --target training -t credit-risk-ai:training .
docker build --target inference -t credit-risk-ai:inference .
```

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ disk space

### Development Environment

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f api

# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
# Jupyter: http://localhost:8888
```

### Production Environment

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

## 📁 File Structure

```
├── Dockerfile                      # Multi-stage container definition
├── docker-compose.yml             # Development environment
├── docker-compose.prod.yml        # Production environment
├── docker-compose.override.yml    # Development overrides
├── docker-compose.security.yml    # Security hardening
├── .dockerignore                  # Build context exclusions
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── nginx/
│   ├── nginx.conf                 # Development nginx config
│   └── nginx.prod.conf           # Production nginx config
├── monitoring/
│   ├── prometheus.yml            # Development monitoring
│   └── prometheus.prod.yml       # Production monitoring
└── scripts/
    ├── security-scan.sh          # Container security scanning
    └── container-hardening.sh    # Security hardening
```

## 🔧 Configuration

### Environment Variables

Create `.env` file for local development:

```bash
# Database
DATABASE_URL=postgresql://postgres:password@postgres:5432/credit_risk_ai
REDIS_URL=redis://redis:6379/0

# API
SECRET_KEY=your-secret-key-here
DEBUG=true
LOG_LEVEL=INFO

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Security
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key
```

### Service Configuration

#### API Service
- **Port**: 8000
- **Health Check**: `/health`
- **Metrics**: `/metrics`
- **Documentation**: `/docs`

#### Database (PostgreSQL)
- **Port**: 5432
- **Database**: `credit_risk_ai`
- **Health Check**: `pg_isready`

#### Cache (Redis)
- **Port**: 6379
- **Health Check**: `redis-cli ping`

#### Monitoring (Prometheus)
- **Port**: 9090
- **Config**: `monitoring/prometheus.yml`

#### Dashboards (Grafana)
- **Port**: 3000
- **Default Login**: admin/admin

## 🛡️ Security

### Container Security Features

- **Non-root user**: All containers run as `appuser` (UID 1000)
- **Read-only filesystem**: Production containers use read-only root
- **Security profiles**: AppArmor and seccomp profiles applied
- **Capability dropping**: Minimal capabilities granted
- **Resource limits**: CPU and memory limits enforced
- **Health checks**: All services have health monitoring

### Security Hardening

```bash
# Apply security hardening
./scripts/container-hardening.sh

# Run security scan
./scripts/security-scan.sh

# Start with security hardening
docker-compose -f docker-compose.yml -f docker-compose.security.yml up -d
```

### Security Scanning

The system includes automated security scanning:

```bash
# Vulnerability scanning with Trivy
trivy image credit-risk-ai:latest

# Docker Bench Security
docker run --rm --net host --pid host --userns host --cap-add audit_control \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  docker/docker-bench-security
```

## 📊 Monitoring

### Metrics Collection

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Application metrics**: Custom business metrics
- **Infrastructure metrics**: Container and system metrics

### Log Aggregation

- **Structured logging**: JSON format with correlation IDs
- **Log rotation**: Automatic log rotation and cleanup
- **Centralized logging**: ELK stack for production

### Health Monitoring

All services include comprehensive health checks:

```bash
# Check service health
docker-compose ps
curl http://localhost:8000/health
```

## 🔄 CI/CD Integration

### Build Pipeline

```yaml
# Example GitHub Actions workflow
name: Build and Test
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build --target production -t credit-risk-ai:${{ github.sha }} .
      - name: Run security scan
        run: ./scripts/security-scan.sh
      - name: Run tests
        run: docker-compose run --rm api pytest
```

### Deployment Pipeline

```yaml
# Example deployment workflow
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d --remove-orphans
```

## 🚀 Deployment Strategies

### Blue-Green Deployment

```bash
# Deploy new version (green)
docker-compose -f docker-compose.prod.yml up -d --scale api=6

# Switch traffic
# Update load balancer configuration

# Remove old version (blue)
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Rolling Updates

```bash
# Update with rolling deployment
docker-compose -f docker-compose.prod.yml up -d --no-deps api
```

### Canary Deployment

```bash
# Deploy canary version
docker-compose -f docker-compose.canary.yml up -d

# Monitor metrics and gradually increase traffic
```

## 🔍 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs api

# Check resource usage
docker stats

# Inspect container
docker inspect <container_id>
```

#### Database Connection Issues
```bash
# Check database health
docker-compose exec postgres pg_isready

# Check network connectivity
docker-compose exec api ping postgres
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check application metrics
curl http://localhost:8000/metrics

# Review logs for errors
docker-compose logs --tail=100 api
```

### Debug Mode

```bash
# Start in debug mode
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Access container shell
docker-compose exec api bash

# Run interactive Python
docker-compose exec api python
```

## 📈 Performance Optimization

### Image Optimization

- **Multi-stage builds**: Minimize final image size
- **Layer caching**: Optimize Dockerfile layer order
- **Base image selection**: Use minimal base images
- **Dependency management**: Pin versions and minimize dependencies

### Runtime Optimization

- **Resource limits**: Set appropriate CPU and memory limits
- **Health checks**: Optimize health check intervals
- **Connection pooling**: Configure database connection pools
- **Caching**: Implement Redis caching strategies

### Scaling Strategies

```bash
# Horizontal scaling
docker-compose -f docker-compose.prod.yml up -d --scale api=5

# Load balancing
# Configure nginx upstream servers

# Database scaling
# Implement read replicas
```

## 🔐 Secrets Management

### Development
```bash
# Use .env file for development
cp .env.example .env
# Edit .env with your values
```

### Production
```bash
# Use Docker secrets
echo "my-secret" | docker secret create db_password -

# Or use external secret management
# - HashiCorp Vault
# - AWS Secrets Manager
# - Azure Key Vault
```

## 📋 Maintenance

### Regular Tasks

```bash
# Update base images
docker-compose pull

# Clean up unused resources
docker system prune -a

# Backup data
docker-compose exec postgres pg_dump -U postgres credit_risk_ai > backup.sql

# Update dependencies
pip-compile requirements.in
```

### Monitoring Tasks

```bash
# Check disk usage
docker system df

# Monitor logs
docker-compose logs --tail=100 -f

# Review security scan results
./scripts/security-scan.sh
```

## 🆘 Support

### Getting Help

1. Check the logs: `docker-compose logs`
2. Review health checks: `docker-compose ps`
3. Check resource usage: `docker stats`
4. Consult troubleshooting guide above
5. Review security checklist: `SECURITY_CHECKLIST.md`

### Useful Commands

```bash
# Complete environment reset
docker-compose down -v --remove-orphans
docker system prune -a

# Backup and restore
docker-compose exec postgres pg_dump -U postgres credit_risk_ai > backup.sql
docker-compose exec -T postgres psql -U postgres credit_risk_ai < backup.sql

# Performance monitoring
docker stats --no-stream
docker-compose top
```

---

For more detailed information, refer to the individual configuration files and the security checklist.