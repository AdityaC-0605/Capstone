#!/bin/bash

# Kubernetes Deployment Validation Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="credit-risk-ai"
TIMEOUT="300s"

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

# Function to check if namespace exists
check_namespace() {
    print_info "Checking namespace: $NAMESPACE"
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_status "Namespace $NAMESPACE exists"
    else
        print_error "Namespace $NAMESPACE does not exist"
        return 1
    fi
}

# Function to check pod status
check_pods() {
    print_info "Checking pod status..."
    
    # Get all pods in namespace
    pods=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name")
    
    for pod in $pods; do
        status=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.phase}')
        ready=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        
        if [[ "$status" == "Running" && "$ready" == "True" ]]; then
            print_status "Pod $pod is running and ready"
        else
            print_error "Pod $pod is not ready (Status: $status, Ready: $ready)"
            kubectl describe pod $pod -n $NAMESPACE | tail -10
        fi
    done
}

# Function to check services
check_services() {
    print_info "Checking services..."
    
    services=("api-service" "postgres-service" "redis-service" "mlflow-service" "federated-server-service" "prometheus-service" "grafana-service")
    
    for service in "${services[@]}"; do
        if kubectl get service $service -n $NAMESPACE &> /dev/null; then
            endpoints=$(kubectl get endpoints $service -n $NAMESPACE -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
            if [[ $endpoints -gt 0 ]]; then
                print_status "Service $service has $endpoints endpoint(s)"
            else
                print_warning "Service $service has no endpoints"
            fi
        else
            print_error "Service $service not found"
        fi
    done
}

# Function to check persistent volumes
check_storage() {
    print_info "Checking persistent volumes..."
    
    pvcs=("postgres-pvc" "redis-pvc" "mlflow-artifacts-pvc" "model-storage-pvc")
    
    for pvc in "${pvcs[@]}"; do
        status=$(kubectl get pvc $pvc -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
        if [[ "$status" == "Bound" ]]; then
            print_status "PVC $pvc is bound"
        else
            print_error "PVC $pvc is not bound (Status: $status)"
        fi
    done
}

# Function to check ingress
check_ingress() {
    print_info "Checking ingress..."
    
    if kubectl get ingress credit-risk-ingress -n $NAMESPACE &> /dev/null; then
        hosts=$(kubectl get ingress credit-risk-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[*].host}')
        print_status "Ingress configured for hosts: $hosts"
    else
        print_warning "Ingress not found"
    fi
}

# Function to check HPA
check_hpa() {
    print_info "Checking Horizontal Pod Autoscaler..."
    
    if kubectl get hpa api-hpa -n $NAMESPACE &> /dev/null; then
        current=$(kubectl get hpa api-hpa -n $NAMESPACE -o jsonpath='{.status.currentReplicas}')
        desired=$(kubectl get hpa api-hpa -n $NAMESPACE -o jsonpath='{.status.desiredReplicas}')
        print_status "HPA api-hpa: Current=$current, Desired=$desired"
    else
        print_warning "HPA not found"
    fi
}

# Function to test API endpoints
test_api_endpoints() {
    print_info "Testing API endpoints..."
    
    # Port forward to API service
    kubectl port-forward svc/api-service 8080:8000 -n $NAMESPACE &
    PF_PID=$!
    sleep 5
    
    # Test health endpoint
    if curl -s http://localhost:8080/health > /dev/null; then
        print_status "Health endpoint is accessible"
    else
        print_error "Health endpoint is not accessible"
    fi
    
    # Test ready endpoint
    if curl -s http://localhost:8080/ready > /dev/null; then
        print_status "Ready endpoint is accessible"
    else
        print_error "Ready endpoint is not accessible"
    fi
    
    # Test docs endpoint
    if curl -s http://localhost:8080/docs > /dev/null; then
        print_status "API documentation is accessible"
    else
        print_warning "API documentation is not accessible"
    fi
    
    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
}

# Function to test database connectivity
test_database() {
    print_info "Testing database connectivity..."
    
    # Test PostgreSQL connection
    if kubectl exec deployment/postgres-deployment -n $NAMESPACE -- pg_isready -U postgres > /dev/null 2>&1; then
        print_status "PostgreSQL is ready"
    else
        print_error "PostgreSQL is not ready"
    fi
    
    # Test Redis connection
    if kubectl exec deployment/redis-deployment -n $NAMESPACE -- redis-cli ping > /dev/null 2>&1; then
        print_status "Redis is ready"
    else
        print_error "Redis is not ready"
    fi
}

# Function to check resource usage
check_resources() {
    print_info "Checking resource usage..."
    
    # Get resource usage for all pods
    kubectl top pods -n $NAMESPACE --no-headers 2>/dev/null | while read line; do
        pod=$(echo $line | awk '{print $1}')
        cpu=$(echo $line | awk '{print $2}')
        memory=$(echo $line | awk '{print $3}')
        print_info "Pod $pod: CPU=$cpu, Memory=$memory"
    done
}

# Function to check logs for errors
check_logs() {
    print_info "Checking recent logs for errors..."
    
    pods=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name")
    
    for pod in $pods; do
        error_count=$(kubectl logs $pod -n $NAMESPACE --tail=100 2>/dev/null | grep -i error | wc -l)
        if [[ $error_count -gt 0 ]]; then
            print_warning "Pod $pod has $error_count error(s) in recent logs"
        else
            print_status "Pod $pod has no recent errors"
        fi
    done
}

# Function to validate security
check_security() {
    print_info "Checking security configuration..."
    
    # Check if pods are running as non-root
    pods=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name")
    
    for pod in $pods; do
        containers=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.spec.containers[*].name}')
        for container in $containers; do
            security_context=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath="{.spec.containers[?(@.name=='$container')].securityContext.runAsNonRoot}")
            if [[ "$security_context" == "true" ]]; then
                print_status "Container $container in pod $pod runs as non-root"
            else
                print_warning "Container $container in pod $pod may be running as root"
            fi
        done
    done
    
    # Check network policies
    if kubectl get networkpolicy -n $NAMESPACE &> /dev/null; then
        policy_count=$(kubectl get networkpolicy -n $NAMESPACE --no-headers | wc -l)
        print_status "Found $policy_count network policy/policies"
    else
        print_warning "No network policies found"
    fi
}

# Function to generate validation report
generate_report() {
    print_info "Generating validation report..."
    
    report_file="validation-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "Kubernetes Deployment Validation Report"
        echo "========================================"
        echo "Timestamp: $(date)"
        echo "Namespace: $NAMESPACE"
        echo ""
        
        echo "Pod Status:"
        kubectl get pods -n $NAMESPACE
        echo ""
        
        echo "Service Status:"
        kubectl get services -n $NAMESPACE
        echo ""
        
        echo "PVC Status:"
        kubectl get pvc -n $NAMESPACE
        echo ""
        
        echo "Ingress Status:"
        kubectl get ingress -n $NAMESPACE
        echo ""
        
        echo "HPA Status:"
        kubectl get hpa -n $NAMESPACE
        echo ""
        
        echo "Resource Usage:"
        kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Metrics server not available"
        echo ""
        
        echo "Recent Events:"
        kubectl get events -n $NAMESPACE --sort-by=.metadata.creationTimestamp | tail -20
        
    } > $report_file
    
    print_status "Validation report saved to: $report_file"
}

# Main validation function
main() {
    print_info "Starting Kubernetes deployment validation..."
    echo ""
    
    check_namespace
    check_pods
    check_services
    check_storage
    check_ingress
    check_hpa
    test_database
    test_api_endpoints
    check_resources
    check_logs
    check_security
    
    echo ""
    print_info "Validation completed!"
    
    # Generate report if requested
    if [[ "${1:-}" == "--report" ]]; then
        generate_report
    fi
}

# Parse command line arguments
case "${1:-validate}" in
    "validate")
        main
        ;;
    "report")
        main --report
        ;;
    "pods")
        check_pods
        ;;
    "services")
        check_services
        ;;
    "api")
        test_api_endpoints
        ;;
    "db")
        test_database
        ;;
    "resources")
        check_resources
        ;;
    "security")
        check_security
        ;;
    *)
        echo "Usage: $0 [validate|report|pods|services|api|db|resources|security]"
        exit 1
        ;;
esac