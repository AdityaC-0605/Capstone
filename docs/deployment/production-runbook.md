# Production Deployment Runbook

## Overview

This runbook provides step-by-step procedures for deploying, monitoring, and maintaining the Sustainable Credit Risk AI system in production environments.

## Table of Contents

1. [Pre-deployment Checklist](#pre-deployment-checklist)
2. [Deployment Procedures](#deployment-procedures)
3. [Post-deployment Validation](#post-deployment-validation)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Rollback Procedures](#rollback-procedures)
7. [Maintenance Tasks](#maintenance-tasks)
8. [Emergency Procedures](#emergency-procedures)

## Pre-deployment Checklist

### Infrastructure Requirements

- [ ] **Kubernetes Cluster**: Version 1.24+ with GPU support
- [ ] **Node Resources**: Minimum 16 CPU cores, 64GB RAM per node
- [ ] **Storage**: 500GB SSD storage with backup capabilities
- [ ] **Network**: Load balancer with SSL termination
- [ ] **Monitoring**: Prometheus, Grafana, and Jaeger deployed
- [ ] **Logging**: ELK stack or equivalent log aggregation

### Security Requirements

- [ ] **SSL Certificates**: Valid certificates for all domains
- [ ] **API Keys**: Generated and securely stored
- [ ] **Database Credentials**: Encrypted and rotated
- [ ] **Network Policies**: Configured for service isolation
- [ ] **RBAC**: Role-based access control configured
- [ ] **Secrets Management**: HashiCorp Vault or equivalent

### Model Validation

- [ ] **Model Performance**: AUC-ROC ≥ 0.85, Accuracy ≥ 0.80
- [ ] **Fairness Metrics**: All bias metrics within acceptable thresholds
- [ ] **Sustainability**: Energy consumption within targets
- [ ] **Explainability**: SHAP and LIME explanations working
- [ ] **Compliance**: FCRA, ECOA, GDPR compliance validated

### Testing Requirements

- [ ] **Unit Tests**: 90%+ code coverage
- [ ] **Integration Tests**: All API endpoints tested
- [ ] **Load Tests**: Performance under expected traffic
- [ ] **Security Tests**: Vulnerability scanning completed
- [ ] **Chaos Engineering**: System resilience validated

## Deployment Procedures

### 1. Environment Preparation

```bash
# Set environment variables
export ENVIRONMENT=production
export NAMESPACE=credit-risk-ai
export IMAGE_TAG=v1.2.0

# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Create namespace if not exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Deploy Infrastructure Services

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml -n $NAMESPACE
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s

# Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml -n $NAMESPACE
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# Deploy MLflow
kubectl apply -f k8s/mlflow-deployment.yaml -n $NAMESPACE
kubectl wait --for=condition=ready pod -l app=mlflow -n $NAMESPACE --timeout=300s
```

### 3. Deploy Application Services

```bash
# Update image tags
sed -i "s|sustainable-credit-risk-ai:production|sustainable-credit-risk-ai:$IMAGE_TAG|g" k8s/api-deployment.yaml

# Deploy API service
kubectl apply -f k8s/api-deployment.yaml -n $NAMESPACE
kubectl rollout status deployment/api-deployment -n $NAMESPACE --timeout=600s

# Deploy Federated Learning Server
kubectl apply -f k8s/federated-server-deployment.yaml -n $NAMESPACE
kubectl rollout status deployment/federated-server-deployment -n $NAMESPACE --timeout=600s
```

### 4. Configure Ingress and Load Balancing

```bash
# Deploy ingress controller
kubectl apply -f k8s/ingress.yaml -n $NAMESPACE

# Verify ingress configuration
kubectl get ingress -n $NAMESPACE
kubectl describe ingress credit-risk-ingress -n $NAMESPACE
```

### 5. Deploy Monitoring Stack

```bash
# Deploy monitoring services
./monitoring/setup-monitoring.sh

# Verify monitoring deployment
kubectl get pods -n monitoring
kubectl get services -n monitoring
```

## Post-deployment Validation

### 1. Health Checks

```bash
# Check all pods are running
kubectl get pods -n $NAMESPACE

# Verify service endpoints
kubectl get services -n $NAMESPACE

# Test health endpoints
curl -f https://api.credit-risk-ai.example.com/health
curl -f https://api.credit-risk-ai.example.com/ready
```

### 2. Functional Testing

```bash
# Run smoke tests
python scripts/smoke_tests.py --url https://api.credit-risk-ai.example.com

# Run integration tests
pytest tests/integration/ --production-url=https://api.credit-risk-ai.example.com

# Validate model performance
python scripts/production_validation.py https://api.credit-risk-ai.example.com
```

### 3. Performance Validation

```bash
# Run load tests
locust -f tests/load/locustfile.py \
  --host=https://api.credit-risk-ai.example.com \
  --users=100 --spawn-rate=10 --run-time=5m \
  --html=load-test-report.html --headless

# Validate performance thresholds
python scripts/check_performance_thresholds.py load-test-report.html
```

### 4. Security Validation

```bash
# Run security scans
trivy image sustainable-credit-risk-ai:$IMAGE_TAG

# Test API security
python tests/security/test_api_security.py --url https://api.credit-risk-ai.example.com

# Validate SSL configuration
sslyze api.credit-risk-ai.example.com
```

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- **Request Rate**: Target > 100 RPS
- **Error Rate**: Target < 1%
- **Latency P95**: Target < 200ms
- **Model Accuracy**: Target > 85%

#### Infrastructure Metrics
- **CPU Utilization**: Alert > 80%
- **Memory Usage**: Alert > 85%
- **Disk Usage**: Alert > 90%
- **Pod Restart Rate**: Alert > 0

#### Business Metrics
- **Prediction Volume**: Monitor trends
- **Approval Rate**: Monitor for anomalies
- **Bias Metrics**: Alert on violations
- **Carbon Footprint**: Monitor sustainability targets

### Alert Configuration

```yaml
# Critical Alerts (Immediate Response)
- High Error Rate (>5% for 2 minutes)
- API Down (Health check failures)
- Model Accuracy Drop (<80%)
- Bias Violation (Any protected attribute)
- Security Breach (Unauthorized access)

# Warning Alerts (Response within 1 hour)
- High Latency (P95 > 200ms for 5 minutes)
- Resource Usage (CPU/Memory > 80%)
- Model Drift (Drift score > 0.1)
- High Carbon Footprint (>5 kg CO2e/hour)

# Info Alerts (Response within 24 hours)
- Deployment Completed
- Scheduled Maintenance
- Performance Degradation
- Configuration Changes
```

### Monitoring Dashboards

1. **System Overview**: Request rates, error rates, latency, accuracy
2. **Infrastructure**: CPU, memory, disk, network usage
3. **Model Performance**: Accuracy, drift, bias metrics
4. **Sustainability**: Energy consumption, carbon emissions
5. **Business Metrics**: Approval rates, processing volume

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. High Error Rate

**Symptoms:**
- HTTP 5xx errors increasing
- Error rate > 5%
- User complaints about failures

**Investigation Steps:**
```bash
# Check pod status
kubectl get pods -n $NAMESPACE

# Check application logs
kubectl logs -f deployment/api-deployment -n $NAMESPACE

# Check resource usage
kubectl top pods -n $NAMESPACE

# Check database connectivity
kubectl exec deployment/api-deployment -n $NAMESPACE -- curl postgres-service:5432
```

**Common Causes:**
- Database connection issues
- Resource exhaustion
- Model loading failures
- Configuration errors

**Resolution:**
1. Scale up resources if needed
2. Restart failed pods
3. Check database connectivity
4. Validate configuration

#### 2. High Latency

**Symptoms:**
- P95 latency > 200ms
- Slow response times
- Timeout errors

**Investigation Steps:**
```bash
# Check resource utilization
kubectl top pods -n $NAMESPACE

# Analyze request patterns
curl -s https://api.credit-risk-ai.example.com/metrics | grep http_request_duration

# Check database performance
kubectl exec deployment/postgres-deployment -n $NAMESPACE -- pg_stat_activity
```

**Resolution:**
1. Scale horizontally (increase replicas)
2. Optimize database queries
3. Enable caching
4. Review model inference time

#### 3. Model Performance Degradation

**Symptoms:**
- Accuracy dropping below 85%
- Increasing bias metrics
- Model drift alerts

**Investigation Steps:**
```bash
# Check model metrics
curl -s https://api.credit-risk-ai.example.com/models/ensemble/metrics

# Analyze recent predictions
python scripts/analyze_model_performance.py --days=7

# Check data drift
python scripts/detect_data_drift.py --baseline=model_training_data.csv
```

**Resolution:**
1. Retrain model with recent data
2. Adjust bias mitigation parameters
3. Update feature engineering
4. Rollback to previous model version

#### 4. Resource Exhaustion

**Symptoms:**
- Pods being killed (OOMKilled)
- CPU throttling
- Slow response times

**Investigation Steps:**
```bash
# Check resource limits
kubectl describe pod <pod-name> -n $NAMESPACE

# Monitor resource usage
kubectl top pods -n $NAMESPACE --sort-by=memory
kubectl top pods -n $NAMESPACE --sort-by=cpu

# Check node resources
kubectl top nodes
```

**Resolution:**
1. Increase resource limits
2. Scale horizontally
3. Optimize memory usage
4. Add more nodes to cluster

## Rollback Procedures

### Automated Rollback

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/api-deployment -n $NAMESPACE

# Check rollback status
kubectl rollout status deployment/api-deployment -n $NAMESPACE

# Verify rollback success
kubectl get pods -n $NAMESPACE
curl -f https://api.credit-risk-ai.example.com/health
```

### Manual Rollback

```bash
# Identify previous version
kubectl rollout history deployment/api-deployment -n $NAMESPACE

# Rollback to specific revision
kubectl rollout undo deployment/api-deployment --to-revision=2 -n $NAMESPACE

# Update ingress if needed
kubectl apply -f k8s/ingress-previous.yaml -n $NAMESPACE
```

### Database Rollback

```bash
# Restore database from backup
kubectl exec deployment/postgres-deployment -n $NAMESPACE -- \
  psql -U postgres -d credit_risk_ai < backup_$(date -d yesterday +%Y%m%d).sql

# Verify data integrity
kubectl exec deployment/postgres-deployment -n $NAMESPACE -- \
  psql -U postgres -d credit_risk_ai -c "SELECT COUNT(*) FROM applications;"
```

## Maintenance Tasks

### Daily Tasks

- [ ] Check system health dashboards
- [ ] Review error logs and alerts
- [ ] Monitor resource usage trends
- [ ] Validate backup completion
- [ ] Check security alerts

### Weekly Tasks

- [ ] Review performance metrics
- [ ] Analyze model performance trends
- [ ] Check bias and fairness metrics
- [ ] Update security patches
- [ ] Review capacity planning

### Monthly Tasks

- [ ] Model retraining with new data
- [ ] Security vulnerability assessment
- [ ] Performance optimization review
- [ ] Disaster recovery testing
- [ ] Documentation updates

### Quarterly Tasks

- [ ] Comprehensive model audit
- [ ] Infrastructure capacity review
- [ ] Security penetration testing
- [ ] Compliance audit
- [ ] Business continuity testing

## Emergency Procedures

### System Outage

1. **Immediate Response (0-5 minutes)**
   - Acknowledge alerts
   - Check system status
   - Notify stakeholders
   - Activate incident response team

2. **Assessment (5-15 minutes)**
   - Identify root cause
   - Assess impact scope
   - Determine recovery strategy
   - Communicate status updates

3. **Recovery (15+ minutes)**
   - Execute recovery procedures
   - Monitor system restoration
   - Validate functionality
   - Document incident

### Security Incident

1. **Immediate Response**
   - Isolate affected systems
   - Preserve evidence
   - Notify security team
   - Activate incident response plan

2. **Investigation**
   - Analyze security logs
   - Identify attack vectors
   - Assess data exposure
   - Document findings

3. **Recovery**
   - Patch vulnerabilities
   - Restore from clean backups
   - Update security controls
   - Monitor for reoccurrence

### Data Breach

1. **Immediate Response**
   - Stop data exposure
   - Preserve evidence
   - Notify legal team
   - Activate breach response plan

2. **Assessment**
   - Identify affected data
   - Assess breach scope
   - Determine notification requirements
   - Document incident

3. **Notification**
   - Notify regulatory authorities
   - Inform affected customers
   - Coordinate with legal team
   - Provide regular updates

## Contact Information

### On-Call Rotation

- **Primary On-Call**: +1-555-ONCALL-1
- **Secondary On-Call**: +1-555-ONCALL-2
- **Escalation Manager**: +1-555-ESCALATE

### Team Contacts

- **DevOps Team**: devops@credit-risk-ai.example.com
- **ML Engineering**: ml-eng@credit-risk-ai.example.com
- **Security Team**: security@credit-risk-ai.example.com
- **Compliance Team**: compliance@credit-risk-ai.example.com

### External Contacts

- **Cloud Provider Support**: [Provider-specific contact]
- **Security Vendor**: [Vendor-specific contact]
- **Legal Counsel**: legal@credit-risk-ai.example.com

## Documentation Links

- **API Documentation**: https://docs.credit-risk-ai.example.com/api
- **Architecture Guide**: https://docs.credit-risk-ai.example.com/architecture
- **Security Policies**: https://docs.credit-risk-ai.example.com/security
- **Compliance Guide**: https://docs.credit-risk-ai.example.com/compliance

**Last Updated:** 2024-01-15  
**Version:** 1.2  
**Next Review:** 2024-04-15