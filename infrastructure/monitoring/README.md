# Monitoring and Observability

This directory contains the complete monitoring and observability stack for the Sustainable Credit Risk AI System.

## Overview

The monitoring system provides comprehensive observability across all system components including:

- **Metrics Collection**: Prometheus with custom metrics for ML models, sustainability, and compliance
- **Visualization**: Grafana dashboards for system overview, sustainability, and fairness monitoring
- **Distributed Tracing**: Jaeger for request tracing across microservices
- **Log Aggregation**: Fluentd + Elasticsearch for centralized logging
- **Alerting**: Prometheus Alertmanager with multi-channel notifications

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Prometheus    │    │     Grafana     │
│   Metrics       │───▶│   (Metrics)     │───▶│  (Dashboards)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Jaeger        │    │  Alertmanager   │
│   Traces        │───▶│   (Tracing)     │    │   (Alerts)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Fluentd       │    │   Slack/Email   │
│   Logs          │───▶│ (Log Collector) │    │ (Notifications) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Elasticsearch  │
                       │ (Log Storage)   │
                       └─────────────────┘
```

## Quick Start

### 1. Deploy Monitoring Stack

```bash
# Deploy all monitoring components
./setup-monitoring.sh

# Or deploy individual components
kubectl apply -f prometheus.yml
kubectl apply -f grafana/
kubectl apply -f jaeger/
kubectl apply -f logging/
```

### 2. Access Monitoring Services

- **Grafana**: http://monitoring.credit-risk-ai.example.com/grafana
  - Username: `admin`
  - Password: `admin123`

- **Prometheus**: http://monitoring.credit-risk-ai.example.com/prometheus
- **Jaeger**: http://monitoring.credit-risk-ai.example.com/jaeger
- **Alertmanager**: http://monitoring.credit-risk-ai.example.com/alertmanager

### 3. Configure Alerts

Update `alert_rules.yml` with your notification channels:

```yaml
# Slack notifications
slack_configs:
- api_url: 'YOUR_SLACK_WEBHOOK_URL'
  channel: '#alerts'

# Email notifications  
email_configs:
- to: 'alerts@your-company.com'
```

## Metrics

### Application Metrics

| Metric | Description | Labels |
|--------|-------------|--------|
| `http_requests_total` | Total HTTP requests | `method`, `status`, `endpoint` |
| `http_request_duration_seconds` | Request duration histogram | `method`, `endpoint` |
| `predictions_total` | Total predictions made | `model_type`, `outcome` |
| `prediction_accuracy_score` | Model accuracy score | `model_name` |
| `model_drift_score` | Model drift detection score | `model_name` |
| `bias_metric` | Bias detection metric | `protected_attribute` |
| `energy_consumption_kwh` | Energy consumption | `component` |
| `carbon_emissions_kg` | Carbon emissions | `region` |

### Infrastructure Metrics

| Metric | Description | Labels |
|--------|-------------|--------|
| `container_cpu_usage_seconds_total` | Container CPU usage | `container`, `pod` |
| `container_memory_usage_bytes` | Container memory usage | `container`, `pod` |
| `kube_pod_status_phase` | Pod status | `pod`, `namespace` |
| `node_filesystem_avail_bytes` | Available disk space | `device`, `mountpoint` |

### Business Metrics

| Metric | Description | Labels |
|--------|-------------|--------|
| `loan_applications_total` | Total loan applications | `status` |
| `loan_approvals_total` | Total loan approvals | `risk_category` |
| `default_rate` | Loan default rate | `time_period` |
| `processing_cost_total` | Processing cost | `component` |

## Dashboards

### 1. System Overview Dashboard
- API request rate and latency
- Error rates and success rates
- Model performance metrics
- Infrastructure resource usage

### 2. Sustainability Dashboard
- Real-time carbon emissions
- Energy consumption trends
- Energy efficiency metrics
- ESG score tracking
- Renewable energy usage

### 3. Fairness & Compliance Dashboard
- Bias detection metrics
- Fairness violation alerts
- Demographic parity tracking
- Compliance status indicators
- Audit trail visualization

### 4. Model Performance Dashboard
- Prediction accuracy trends
- Model drift detection
- Feature importance changes
- Training performance metrics

## Alerting Rules

### Critical Alerts
- **API High Error Rate**: Error rate > 5% for 2 minutes
- **Model Accuracy Drop**: Accuracy < 80% for 5 minutes
- **Bias Violation**: Bias metric > 0.1 for any protected attribute
- **High Carbon Footprint**: Emissions > 5 kg CO2e/hour for 10 minutes

### Warning Alerts
- **API High Latency**: P95 latency > 200ms for 5 minutes
- **Model Drift**: Drift score > 0.1 for 1 minute
- **High Resource Usage**: CPU/Memory > 90% for 5 minutes
- **Federated Client Disconnection**: < 80% clients connected

### Infrastructure Alerts
- **Pod Crash Looping**: Pod restarting frequently
- **Disk Space Low**: < 10% disk space available
- **Database Connections High**: > 80 active connections

## Distributed Tracing

### Trace Collection
Jaeger collects traces from:
- API requests end-to-end
- Model inference pipelines
- Federated learning communications
- Data processing workflows

### Trace Analysis
- Request flow visualization
- Performance bottleneck identification
- Error propagation tracking
- Service dependency mapping

## Log Aggregation

### Log Sources
- Application logs (API, ML services)
- Kubernetes system logs
- Audit logs (compliance events)
- Security logs (authentication, authorization)

### Log Processing
- PII anonymization
- Structured log parsing
- Log enrichment with metadata
- Security event filtering

### Log Storage
- Elasticsearch cluster for scalable storage
- Index rotation and retention policies
- Full-text search capabilities
- Log correlation and analysis

## Configuration

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'credit-risk-api'
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: api-service
```

### Grafana Provisioning
```yaml
# datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus-service:9090
    isDefault: true
```

### Fluentd Configuration
```ruby
# fluent.conf
<source>
  @type tail
  path /var/log/containers/*.log
  tag kubernetes.*
  format json
</source>

<match kubernetes.**>
  @type elasticsearch
  host elasticsearch-service
  port 9200
  index_name kubernetes-logs
</match>
```

## Maintenance

### Regular Tasks
- Review and update alert thresholds
- Rotate log indices and clean old data
- Update dashboard configurations
- Monitor storage usage and capacity

### Troubleshooting
- Check service health endpoints
- Verify metric collection
- Validate log ingestion
- Test alert delivery

### Backup and Recovery
- Export Grafana dashboards
- Backup Prometheus configuration
- Archive critical log data
- Document recovery procedures

## Security

### Access Control
- RBAC for Kubernetes resources
- Authentication for monitoring services
- Network policies for service isolation
- Audit logging for administrative actions

### Data Protection
- PII anonymization in logs
- Encrypted communication channels
- Secure credential management
- Regular security updates

## Performance Optimization

### Resource Allocation
- Right-size monitoring components
- Optimize metric retention periods
- Configure efficient log parsing
- Implement metric sampling

### Scaling Considerations
- Horizontal scaling for high load
- Federation for multi-cluster setups
- Load balancing for query distribution
- Caching for frequently accessed data

## Integration

### CI/CD Integration
- Automated dashboard deployment
- Alert rule validation
- Monitoring configuration testing
- Performance regression detection

### External Systems
- ITSM tool integration
- Business intelligence platforms
- Compliance reporting systems
- Cost management tools

## Support

For issues and questions:
1. Check service health dashboards
2. Review alert notifications
3. Examine application logs
4. Contact the monitoring team

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Fluentd Documentation](https://docs.fluentd.org/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/)