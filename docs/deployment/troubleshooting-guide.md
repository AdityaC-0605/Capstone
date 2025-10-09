# Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for the Sustainable Credit Risk AI system, covering common issues, diagnostic steps, and resolution strategies.

## Quick Reference

### Emergency Contacts
- **On-Call Engineer**: +1-555-ONCALL-1
- **System Administrator**: +1-555-SYSADMIN
- **Security Team**: +1-555-SECURITY

### Critical Commands
```bash
# System status
kubectl get pods -n credit-risk-ai
kubectl get services -n credit-risk-ai

# Logs
kubectl logs -f deployment/api-deployment -n credit-risk-ai

# Health checks
curl https://api.credit-risk-ai.example.com/health
```

## Issue Categories

### 1. API Service Issues

#### 1.1 Service Unavailable (HTTP 503)

**Symptoms:**
- API returns 503 Service Unavailable
- Health check endpoints failing
- Load balancer showing no healthy backends

**Diagnostic Steps:**
```bash
# Check pod status
kubectl get pods -l app=api -n credit-risk-ai

# Check pod logs
kubectl logs -f deployment/api-deployment -n credit-risk-ai --tail=100

# Check service endpoints
kubectl get endpoints api-service -n credit-risk-ai

# Check ingress status
kubectl describe ingress credit-risk-ingress -n credit-risk-ai
```

**Common Causes:**
- All pods crashed or restarting
- Resource exhaustion (CPU/Memory)
- Database connectivity issues
- Configuration errors

**Resolution Steps:**
1. **Scale up if resource constrained:**
   ```bash
   kubectl scale deployment api-deployment --replicas=5 -n credit-risk-ai
   ```

2. **Restart failed pods:**
   ```bash
   kubectl delete pods -l app=api -n credit-risk-ai
   ```

3. **Check resource limits:**
   ```bash
   kubectl describe deployment api-deployment -n credit-risk-ai
   ```

4. **Verify database connectivity:**
   ```bash
   kubectl exec deployment/api-deployment -n credit-risk-ai -- \
     curl -f postgres-service:5432 || echo "Database unreachable"
   ```

#### 1.2 High Error Rate (HTTP 5xx)

**Symptoms:**
- Increased 500/502/504 errors
- Error rate > 5%
- User complaints about failures

**Diagnostic Steps:**
```bash
# Check error patterns in logs
kubectl logs deployment/api-deployment -n credit-risk-ai | grep -i error | tail -20

# Check metrics
curl -s https://api.credit-risk-ai.example.com/metrics | grep http_requests_total

# Check database status
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
  pg_isready -U postgres
```

**Common Causes:**
- Model loading failures
- Database connection pool exhaustion
- Memory leaks
- Invalid input data

**Resolution Steps:**
1. **Check model loading:**
   ```bash
   kubectl logs deployment/api-deployment -n credit-risk-ai | grep -i "model"
   ```

2. **Restart application pods:**
   ```bash
   kubectl rollout restart deployment/api-deployment -n credit-risk-ai
   ```

3. **Scale database connections:**
   ```bash
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -c "SHOW max_connections;"
   ```

#### 1.3 High Latency

**Symptoms:**
- P95 latency > 200ms
- Timeout errors
- Slow response times

**Diagnostic Steps:**
```bash
# Check resource utilization
kubectl top pods -n credit-risk-ai

# Analyze request patterns
curl -s https://api.credit-risk-ai.example.com/metrics | \
  grep http_request_duration_seconds

# Check database performance
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
  psql -U postgres -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

**Resolution Steps:**
1. **Scale horizontally:**
   ```bash
   kubectl scale deployment api-deployment --replicas=6 -n credit-risk-ai
   ```

2. **Optimize database queries:**
   ```bash
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
   ```

3. **Enable caching:**
   ```bash
   # Check Redis status
   kubectl exec deployment/redis-deployment -n credit-risk-ai -- redis-cli ping
   ```

### 2. Model Performance Issues

#### 2.1 Model Accuracy Drop

**Symptoms:**
- Accuracy below 85%
- Increasing prediction errors
- Model drift alerts

**Diagnostic Steps:**
```bash
# Check model metrics
curl -s https://api.credit-risk-ai.example.com/models/ensemble/metrics

# Analyze recent predictions
python scripts/analyze_model_performance.py --days=7

# Check for data drift
python scripts/detect_data_drift.py --baseline=training_data.csv --current=recent_data.csv
```

**Resolution Steps:**
1. **Retrain model:**
   ```bash
   kubectl create job model-retrain --from=cronjob/scheduled-training -n credit-risk-ai
   ```

2. **Rollback to previous model:**
   ```bash
   # Update model version in deployment
   kubectl set image deployment/api-deployment \
     api=sustainable-credit-risk-ai:v1.1.0 -n credit-risk-ai
   ```

3. **Adjust model parameters:**
   ```bash
   # Update model configuration
   kubectl patch configmap model-config -n credit-risk-ai --patch='
   data:
     confidence_threshold: "0.7"
     bias_threshold: "0.05"
   '
   ```

#### 2.2 Bias Violations

**Symptoms:**
- Fairness metrics below thresholds
- Disparate impact detected
- Compliance alerts

**Diagnostic Steps:**
```bash
# Check bias metrics
curl -s https://api.credit-risk-ai.example.com/bias/analysis \
  -H "Content-Type: application/json" \
  -d '{"protected_attributes": ["gender", "race", "age_group"]}'

# Analyze recent decisions
python scripts/analyze_bias.py --time-range=24h
```

**Resolution Steps:**
1. **Apply bias mitigation:**
   ```bash
   # Update bias mitigation parameters
   kubectl patch configmap model-config -n credit-risk-ai --patch='
   data:
     bias_mitigation_enabled: "true"
     fairness_constraint_weight: "0.1"
   '
   ```

2. **Retrain with fairness constraints:**
   ```bash
   kubectl create job fair-retrain --from=cronjob/scheduled-training -n credit-risk-ai
   kubectl set env job/fair-retrain FAIRNESS_TRAINING=true -n credit-risk-ai
   ```

### 3. Infrastructure Issues

#### 3.1 Pod Crashes (CrashLoopBackOff)

**Symptoms:**
- Pods in CrashLoopBackOff state
- Frequent pod restarts
- Application unavailable

**Diagnostic Steps:**
```bash
# Check pod status
kubectl get pods -n credit-risk-ai

# Check pod events
kubectl describe pod <pod-name> -n credit-risk-ai

# Check logs from crashed pod
kubectl logs <pod-name> -n credit-risk-ai --previous
```

**Common Causes:**
- Out of memory (OOMKilled)
- Configuration errors
- Missing dependencies
- Health check failures

**Resolution Steps:**
1. **Increase memory limits:**
   ```bash
   kubectl patch deployment api-deployment -n credit-risk-ai --patch='
   spec:
     template:
       spec:
         containers:
         - name: api
           resources:
             limits:
               memory: "4Gi"
             requests:
               memory: "2Gi"
   '
   ```

2. **Fix configuration:**
   ```bash
   # Check configuration
   kubectl get configmap -n credit-risk-ai
   kubectl describe configmap credit-risk-config -n credit-risk-ai
   ```

3. **Update health check parameters:**
   ```bash
   kubectl patch deployment api-deployment -n credit-risk-ai --patch='
   spec:
     template:
       spec:
         containers:
         - name: api
           livenessProbe:
             initialDelaySeconds: 60
             timeoutSeconds: 30
   '
   ```

#### 3.2 Resource Exhaustion

**Symptoms:**
- High CPU/Memory usage
- Slow performance
- Pod evictions

**Diagnostic Steps:**
```bash
# Check resource usage
kubectl top pods -n credit-risk-ai --sort-by=memory
kubectl top nodes

# Check resource limits
kubectl describe deployment api-deployment -n credit-risk-ai

# Check node capacity
kubectl describe nodes
```

**Resolution Steps:**
1. **Scale horizontally:**
   ```bash
   kubectl scale deployment api-deployment --replicas=8 -n credit-risk-ai
   ```

2. **Add more nodes:**
   ```bash
   # For cloud providers, increase node pool size
   # For on-premises, add physical nodes
   ```

3. **Optimize resource usage:**
   ```bash
   # Enable resource quotas
   kubectl apply -f k8s/resource-quotas.yaml -n credit-risk-ai
   ```

#### 3.3 Storage Issues

**Symptoms:**
- Disk space warnings
- Database write failures
- Log ingestion failures

**Diagnostic Steps:**
```bash
# Check PVC usage
kubectl get pvc -n credit-risk-ai

# Check disk usage on nodes
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- df -h

# Check storage class
kubectl get storageclass
```

**Resolution Steps:**
1. **Expand PVC:**
   ```bash
   kubectl patch pvc postgres-pvc -n credit-risk-ai --patch='
   spec:
     resources:
       requests:
         storage: 20Gi
   '
   ```

2. **Clean up old data:**
   ```bash
   # Clean up old logs
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -c "DELETE FROM logs WHERE created_at < NOW() - INTERVAL '30 days';"
   ```

### 4. Database Issues

#### 4.1 Connection Pool Exhaustion

**Symptoms:**
- "Too many connections" errors
- Connection timeouts
- Database unavailable

**Diagnostic Steps:**
```bash
# Check active connections
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Check connection limits
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
  psql -U postgres -c "SHOW max_connections;"

# Check connection pool configuration
kubectl describe configmap postgres-config -n credit-risk-ai
```

**Resolution Steps:**
1. **Increase connection limit:**
   ```bash
   kubectl patch configmap postgres-config -n credit-risk-ai --patch='
   data:
     max_connections: "200"
   '
   kubectl rollout restart deployment/postgres-deployment -n credit-risk-ai
   ```

2. **Optimize connection pooling:**
   ```bash
   # Deploy PgBouncer
   kubectl apply -f k8s/pgbouncer-deployment.yaml -n credit-risk-ai
   ```

#### 4.2 Slow Queries

**Symptoms:**
- High database CPU usage
- Slow API responses
- Query timeouts

**Diagnostic Steps:**
```bash
# Check slow queries
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
  psql -U postgres -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check database locks
kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
  psql -U postgres -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Resolution Steps:**
1. **Add database indexes:**
   ```bash
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -c "CREATE INDEX CONCURRENTLY idx_applications_created_at ON applications(created_at);"
   ```

2. **Optimize queries:**
   ```bash
   # Analyze query plans
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -c "EXPLAIN ANALYZE SELECT * FROM applications WHERE status = 'pending';"
   ```

### 5. Monitoring and Alerting Issues

#### 5.1 Missing Metrics

**Symptoms:**
- Gaps in monitoring dashboards
- Missing alert notifications
- Prometheus targets down

**Diagnostic Steps:**
```bash
# Check Prometheus targets
curl -s http://prometheus-service:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Check service discovery
kubectl get endpoints -n credit-risk-ai

# Check metric endpoints
curl -s https://api.credit-risk-ai.example.com/metrics
```

**Resolution Steps:**
1. **Restart Prometheus:**
   ```bash
   kubectl rollout restart deployment/prometheus -n monitoring
   ```

2. **Fix service annotations:**
   ```bash
   kubectl patch service api-service -n credit-risk-ai --patch='
   metadata:
     annotations:
       prometheus.io/scrape: "true"
       prometheus.io/port: "8000"
       prometheus.io/path: "/metrics"
   '
   ```

#### 5.2 Alert Fatigue

**Symptoms:**
- Too many false positive alerts
- Important alerts missed
- Alert storm conditions

**Resolution Steps:**
1. **Adjust alert thresholds:**
   ```bash
   # Update alert rules
   kubectl patch configmap prometheus-config -n monitoring --patch='
   data:
     alert_rules.yml: |
       groups:
       - name: api_alerts
         rules:
         - alert: HighErrorRate
           expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
           for: 5m
   '
   ```

2. **Implement alert grouping:**
   ```bash
   # Update Alertmanager configuration
   kubectl patch configmap alertmanager-config -n monitoring --patch='
   data:
     alertmanager.yml: |
       route:
         group_by: ["alertname", "cluster", "service"]
         group_wait: 30s
         group_interval: 5m
         repeat_interval: 12h
   '
   ```

## Diagnostic Tools

### Log Analysis

```bash
# Search for errors in logs
kubectl logs deployment/api-deployment -n credit-risk-ai | grep -i error

# Follow logs in real-time
kubectl logs -f deployment/api-deployment -n credit-risk-ai

# Get logs from all pods
kubectl logs -l app=api -n credit-risk-ai --tail=100

# Export logs for analysis
kubectl logs deployment/api-deployment -n credit-risk-ai --since=1h > api-logs.txt
```

### Performance Analysis

```bash
# Check resource usage
kubectl top pods -n credit-risk-ai
kubectl top nodes

# Get detailed resource information
kubectl describe pod <pod-name> -n credit-risk-ai

# Check HPA status
kubectl get hpa -n credit-risk-ai
kubectl describe hpa api-hpa -n credit-risk-ai
```

### Network Diagnostics

```bash
# Test service connectivity
kubectl exec deployment/api-deployment -n credit-risk-ai -- \
  curl -f postgres-service:5432

# Check DNS resolution
kubectl exec deployment/api-deployment -n credit-risk-ai -- \
  nslookup postgres-service

# Test external connectivity
kubectl exec deployment/api-deployment -n credit-risk-ai -- \
  curl -f https://api.github.com
```

## Recovery Procedures

### Service Recovery

1. **Immediate Actions:**
   ```bash
   # Scale up replicas
   kubectl scale deployment api-deployment --replicas=6 -n credit-risk-ai
   
   # Restart failed pods
   kubectl delete pods -l app=api -n credit-risk-ai
   
   # Check service status
   kubectl get pods -n credit-risk-ai
   ```

2. **Validation:**
   ```bash
   # Test health endpoints
   curl -f https://api.credit-risk-ai.example.com/health
   
   # Run smoke tests
   python scripts/smoke_tests.py --url https://api.credit-risk-ai.example.com
   ```

### Database Recovery

1. **Connection Issues:**
   ```bash
   # Restart database
   kubectl rollout restart deployment/postgres-deployment -n credit-risk-ai
   
   # Check connections
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
   ```

2. **Data Corruption:**
   ```bash
   # Restore from backup
   kubectl exec deployment/postgres-deployment -n credit-risk-ai -- \
     psql -U postgres -d credit_risk_ai < backup_latest.sql
   ```

## Prevention Strategies

### Proactive Monitoring

1. **Set up comprehensive alerts**
2. **Regular health checks**
3. **Capacity planning**
4. **Performance baselines**

### Regular Maintenance

1. **Update dependencies**
2. **Rotate credentials**
3. **Clean up old data**
4. **Test disaster recovery**

### Documentation

1. **Keep runbooks updated**
2. **Document known issues**
3. **Maintain contact lists**
4. **Record lessons learned**

## Escalation Procedures

### Level 1: Self-Service
- Use this troubleshooting guide
- Check monitoring dashboards
- Review recent changes

### Level 2: On-Call Engineer
- Contact: +1-555-ONCALL-1
- Provide: Issue description, steps taken, current status

### Level 3: Subject Matter Expert
- ML Engineering: ml-eng@credit-risk-ai.example.com
- Infrastructure: infra@credit-risk-ai.example.com
- Security: security@credit-risk-ai.example.com

### Level 4: Management Escalation
- Engineering Manager: eng-mgr@credit-risk-ai.example.com
- CTO: cto@credit-risk-ai.example.com

## Contact Information

### Emergency Contacts
- **On-Call**: +1-555-ONCALL-1
- **Backup On-Call**: +1-555-ONCALL-2
- **Security Hotline**: +1-555-SECURITY

### Team Contacts
- **DevOps**: devops@credit-risk-ai.example.com
- **ML Engineering**: ml-eng@credit-risk-ai.example.com
- **Platform**: platform@credit-risk-ai.example.com

### External Support
- **Cloud Provider**: [Provider Support Number]
- **Monitoring Vendor**: [Vendor Support Number]

**Last Updated:** 2024-01-15  
**Version:** 1.2  
**Next Review:** 2024-04-15