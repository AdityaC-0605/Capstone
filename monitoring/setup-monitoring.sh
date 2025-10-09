#!/bin/bash

# Monitoring and Observability Setup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="credit-risk-ai"
MONITORING_NAMESPACE="monitoring"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    print_status "kubectl is available"
}

# Function to create namespaces
create_namespaces() {
    print_info "Creating namespaces..."
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace $MONITORING_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    print_status "Namespaces created"
}

# Function to deploy Prometheus
deploy_prometheus() {
    print_info "Deploying Prometheus..."
    
    # Create ConfigMap for Prometheus configuration
    kubectl create configmap prometheus-config \
        --from-file=prometheus.yml \
        --from-file=alert_rules.yml \
        --from-file=recording_rules.yml \
        -n $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.47.0
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus/'
          - '--web.console.libraries=/etc/prometheus/console_libraries'
          - '--web.console.templates=/etc/prometheus/consoles'
          - '--storage.tsdb.retention.time=30d'
          - '--web.enable-lifecycle'
          - '--web.enable-admin-api'
        ports:
        - containerPort: 9090
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
        volumeMounts:
        - name: prometheus-config-volume
          mountPath: /etc/prometheus/
        - name: prometheus-storage-volume
          mountPath: /prometheus/
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          timeoutSeconds: 30
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 30
          timeoutSeconds: 30
      volumes:
      - name: prometheus-config-volume
        configMap:
          defaultMode: 420
          name: prometheus-config
      - name: prometheus-storage-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
spec:
  selector:
    app: prometheus
  type: ClusterIP
  ports:
  - port: 9090
    targetPort: 9090
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: $MONITORING_NAMESPACE
EOF
    
    print_status "Prometheus deployed"
}

# Function to deploy Grafana
deploy_grafana() {
    print_info "Deploying Grafana..."
    
    # Create ConfigMaps for Grafana dashboards
    kubectl create configmap grafana-dashboards \
        --from-file=grafana/dashboards/ \
        -n $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Grafana
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.1.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin123"
        - name: GF_INSTALL_PLUGINS
          value: "grafana-piechart-panel,grafana-worldmap-panel"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 200m
            memory: 512Mi
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-dashboards
          mountPath: /etc/grafana/provisioning/dashboards
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
      volumes:
      - name: grafana-storage
        emptyDir: {}
      - name: grafana-dashboards
        configMap:
          name: grafana-dashboards
---
apiVersion: v1
kind: Service
metadata:
  name: grafana-service
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
spec:
  selector:
    app: grafana
  type: ClusterIP
  ports:
  - port: 3000
    targetPort: 3000
EOF
    
    print_status "Grafana deployed"
}

# Function to deploy Jaeger
deploy_jaeger() {
    print_info "Deploying Jaeger for distributed tracing..."
    
    kubectl apply -f jaeger/jaeger-deployment.yaml -n $MONITORING_NAMESPACE
    
    print_status "Jaeger deployed"
}

# Function to deploy Elasticsearch
deploy_elasticsearch() {
    print_info "Deploying Elasticsearch for log storage..."
    
    kubectl apply -f logging/elasticsearch-deployment.yaml -n $MONITORING_NAMESPACE
    
    # Wait for Elasticsearch to be ready
    print_info "Waiting for Elasticsearch to be ready..."
    kubectl wait --for=condition=ready pod -l app=elasticsearch -n $MONITORING_NAMESPACE --timeout=300s
    
    print_status "Elasticsearch deployed"
}

# Function to deploy Fluentd
deploy_fluentd() {
    print_info "Deploying Fluentd for log aggregation..."
    
    kubectl apply -f logging/fluentd-config.yaml -n $MONITORING_NAMESPACE
    
    print_status "Fluentd deployed"
}

# Function to deploy Node Exporter
deploy_node_exporter() {
    print_info "Deploying Node Exporter..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9100"
    spec:
      hostPID: true
      hostIPC: true
      hostNetwork: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.6.1
        args:
          - --path.sysfs=/host/sys
          - --path.rootfs=/host/root
          - --no-collector.wifi
          - --no-collector.hwmon
          - --collector.filesystem.ignored-mount-points=^/(dev|proc|sys|var/lib/docker/.+|var/lib/kubelet/pods/.+)($|/)
          - --collector.netclass.ignored-devices=^(veth.*)$
        ports:
        - containerPort: 9100
          protocol: TCP
        resources:
          limits:
            cpu: 250m
            memory: 180Mi
          requests:
            cpu: 102m
            memory: 180Mi
        volumeMounts:
        - mountPath: /host/sys
          name: sys
          readOnly: true
        - mountPath: /host/root
          mountPropagation: HostToContainer
          name: root
          readOnly: true
      tolerations:
      - operator: Exists
      volumes:
      - hostPath:
          path: /sys
        name: sys
      - hostPath:
          path: /
        name: root
---
apiVersion: v1
kind: Service
metadata:
  name: node-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: node-exporter
spec:
  type: ClusterIP
  clusterIP: None
  selector:
    app: node-exporter
  ports:
  - name: node-exporter
    port: 9100
    protocol: TCP
EOF
    
    print_status "Node Exporter deployed"
}

# Function to deploy Alertmanager
deploy_alertmanager() {
    print_info "Deploying Alertmanager..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: $MONITORING_NAMESPACE
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alertmanager@credit-risk-ai.example.com'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
      - match:
          service: compliance
        receiver: 'compliance-alerts'
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://webhook-service:5000/alerts'
    
    - name: 'critical-alerts'
      slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#critical-alerts'
        title: 'Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    
    - name: 'compliance-alerts'
      email_configs:
      - to: 'compliance@credit-risk-ai.example.com'
        subject: 'Compliance Alert'
        body: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.26.0
        args:
          - '--config.file=/etc/alertmanager/alertmanager.yml'
          - '--storage.path=/alertmanager'
        ports:
        - containerPort: 9093
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        volumeMounts:
        - name: alertmanager-config-volume
          mountPath: /etc/alertmanager
        - name: alertmanager-storage-volume
          mountPath: /alertmanager
      volumes:
      - name: alertmanager-config-volume
        configMap:
          name: alertmanager-config
      - name: alertmanager-storage-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
spec:
  selector:
    app: alertmanager
  type: ClusterIP
  ports:
  - port: 9093
    targetPort: 9093
EOF
    
    print_status "Alertmanager deployed"
}

# Function to create ingress for monitoring services
create_monitoring_ingress() {
    print_info "Creating ingress for monitoring services..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: $MONITORING_NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - monitoring.credit-risk-ai.example.com
    secretName: monitoring-tls
  rules:
  - host: monitoring.credit-risk-ai.example.com
    http:
      paths:
      - path: /grafana
        pathType: Prefix
        backend:
          service:
            name: grafana-service
            port:
              number: 3000
      - path: /prometheus
        pathType: Prefix
        backend:
          service:
            name: prometheus-service
            port:
              number: 9090
      - path: /jaeger
        pathType: Prefix
        backend:
          service:
            name: jaeger-query
            port:
              number: 16686
      - path: /alertmanager
        pathType: Prefix
        backend:
          service:
            name: alertmanager
            port:
              number: 9093
EOF
    
    print_status "Monitoring ingress created"
}

# Function to validate deployment
validate_deployment() {
    print_info "Validating monitoring deployment..."
    
    # Check if all pods are running
    kubectl get pods -n $MONITORING_NAMESPACE
    
    # Wait for all deployments to be ready
    kubectl wait --for=condition=available deployment --all -n $MONITORING_NAMESPACE --timeout=300s
    
    print_status "All monitoring services are running"
    
    # Display access information
    print_info "Monitoring services access information:"
    echo "  Grafana: http://monitoring.credit-risk-ai.example.com/grafana (admin/admin123)"
    echo "  Prometheus: http://monitoring.credit-risk-ai.example.com/prometheus"
    echo "  Jaeger: http://monitoring.credit-risk-ai.example.com/jaeger"
    echo "  Alertmanager: http://monitoring.credit-risk-ai.example.com/alertmanager"
}

# Main function
main() {
    print_info "Setting up monitoring and observability for Credit Risk AI System"
    echo "=" * 70
    
    check_kubectl
    create_namespaces
    deploy_prometheus
    deploy_grafana
    deploy_jaeger
    deploy_elasticsearch
    deploy_fluentd
    deploy_node_exporter
    deploy_alertmanager
    create_monitoring_ingress
    validate_deployment
    
    print_status "Monitoring and observability setup completed successfully!"
}

# Run main function
main "$@"