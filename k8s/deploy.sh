#!/bin/bash

# Kubernetes Deployment Script for Sustainable Credit Risk AI System
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="credit-risk-ai"
KUBECTL_TIMEOUT="300s"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    print_status "kubectl is available"
}

# Function to check if cluster is accessible
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_status "Connected to Kubernetes cluster"
}

# Function to create namespace
create_namespace() {
    print_status "Creating namespace: $NAMESPACE"
    kubectl apply -f namespace.yaml
    kubectl config set-context --current --namespace=$NAMESPACE
}

# Function to apply configurations
apply_configs() {
    print_status "Applying ConfigMaps and Secrets..."
    kubectl apply -f configmap.yaml
    kubectl apply -f secrets.yaml
    
    print_status "Applying RBAC configurations..."
    kubectl apply -f rbac.yaml
    
    print_status "Creating Persistent Volumes..."
    kubectl apply -f persistent-volumes.yaml
}

# Function to deploy infrastructure services
deploy_infrastructure() {
    print_status "Deploying PostgreSQL..."
    kubectl apply -f postgres-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=postgres --timeout=$KUBECTL_TIMEOUT
    
    print_status "Deploying Redis..."
    kubectl apply -f redis-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=redis --timeout=$KUBECTL_TIMEOUT
    
    print_status "Deploying MLflow..."
    kubectl apply -f mlflow-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=mlflow --timeout=$KUBECTL_TIMEOUT
}

# Function to deploy application services
deploy_application() {
    print_status "Deploying API service..."
    kubectl apply -f api-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=api --timeout=$KUBECTL_TIMEOUT
    
    print_status "Deploying Federated Learning Server..."
    kubectl apply -f federated-server-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=federated-server --timeout=$KUBECTL_TIMEOUT
}

# Function to deploy monitoring
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    kubectl apply -f monitoring-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=prometheus --timeout=$KUBECTL_TIMEOUT
    kubectl wait --for=condition=ready pod -l app=grafana --timeout=$KUBECTL_TIMEOUT
}

# Function to setup ingress
setup_ingress() {
    print_status "Setting up ingress..."
    kubectl apply -f ingress.yaml
}

# Function to run training job
run_training() {
    print_status "Starting training job..."
    kubectl apply -f training-job.yaml
}

# Function to check deployment status
check_status() {
    print_status "Checking deployment status..."
    kubectl get pods -n $NAMESPACE
    kubectl get services -n $NAMESPACE
    kubectl get ingress -n $NAMESPACE
}

# Main deployment function
main() {
    print_status "Starting Kubernetes deployment for Sustainable Credit Risk AI System"
    
    check_kubectl
    check_cluster
    create_namespace
    apply_configs
    deploy_infrastructure
    deploy_application
    deploy_monitoring
    setup_ingress
    
    print_status "Deployment completed successfully!"
    check_status
    
    print_status "Access URLs:"
    echo "  API: https://api.credit-risk-ai.example.com"
    echo "  Monitoring: https://monitoring.credit-risk-ai.example.com/grafana"
    echo "  MLflow: https://mlflow.credit-risk-ai.example.com"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "training")
        run_training
        ;;
    "status")
        check_status
        ;;
    "clean")
        print_warning "Cleaning up deployment..."
        kubectl delete namespace $NAMESPACE
        ;;
    *)
        echo "Usage: $0 [deploy|training|status|clean]"
        exit 1
        ;;
esac