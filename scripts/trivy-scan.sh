#!/bin/bash

# Trivy Security Scan Script for Sustainable Credit Risk AI
# This script provides optimized Trivy scanning with disk space management

set -e

# Configuration
IMAGE_NAME="sustainable-credit-risk-ai"
SCAN_TARGET="${1:-production}"
OUTPUT_DIR="./security-reports"
CACHE_DIR="./.trivycache"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Starting Trivy Security Scan for ${IMAGE_NAME}:${SCAN_TARGET}${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Function to clean up Docker resources
cleanup_docker() {
    echo -e "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
    docker system prune -af --volumes || true
    docker builder prune -af || true
    echo -e "${GREEN}✅ Docker cleanup completed${NC}"
}

# Function to check disk space
check_disk_space() {
    local available_space=$(df /tmp | awk 'NR==2 {print $4}')
    local required_space=2000000  # 2GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo -e "${RED}❌ Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB${NC}"
        cleanup_docker
        return 1
    fi
    
    echo -e "${GREEN}✅ Sufficient disk space available: ${available_space}KB${NC}"
    return 0
}

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}🔨 Building Docker image: ${IMAGE_NAME}:${SCAN_TARGET}${NC}"
    
    if ! docker build --target "$SCAN_TARGET" -t "${IMAGE_NAME}:${SCAN_TARGET}" .; then
        echo -e "${RED}❌ Failed to build Docker image${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
}

# Function to run Trivy scan
run_trivy_scan() {
    local scan_type="$1"
    local output_file="$2"
    local additional_args="$3"
    
    echo -e "${YELLOW}🔍 Running Trivy ${scan_type} scan...${NC}"
    
    # Base Trivy command with optimizations
    local trivy_cmd="trivy image \
        --format table \
        --output ${output_file} \
        --cache-dir ${CACHE_DIR} \
        --skip-version-check \
        --ignore-unfixed \
        --severity CRITICAL,HIGH,MEDIUM \
        ${additional_args} \
        ${IMAGE_NAME}:${SCAN_TARGET}"
    
    if ! eval "$trivy_cmd"; then
        echo -e "${RED}❌ Trivy ${scan_type} scan failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Trivy ${scan_type} scan completed${NC}"
    return 0
}

# Function to run SARIF scan for GitHub integration
run_sarif_scan() {
    local output_file="${OUTPUT_DIR}/trivy-${SCAN_TARGET}-results.sarif"
    
    echo -e "${YELLOW}📊 Running Trivy SARIF scan for GitHub integration...${NC}"
    
    local trivy_cmd="trivy image \
        --format sarif \
        --output ${output_file} \
        --cache-dir ${CACHE_DIR} \
        --skip-version-check \
        --ignore-unfixed \
        --severity CRITICAL,HIGH,MEDIUM \
        --scanners vuln,secret \
        ${IMAGE_NAME}:${SCAN_TARGET}"
    
    if ! eval "$trivy_cmd"; then
        echo -e "${RED}❌ Trivy SARIF scan failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ SARIF report generated: ${output_file}${NC}"
    return 0
}

# Main execution
main() {
    # Check disk space before starting
    if ! check_disk_space; then
        echo -e "${RED}❌ Cannot proceed due to insufficient disk space${NC}"
        exit 1
    fi
    
    # Build the Docker image
    build_image
    
    # Run vulnerability scan
    run_trivy_scan "vulnerability" "${OUTPUT_DIR}/trivy-${SCAN_TARGET}-vuln.txt" "--scanners vuln"
    
    # Run secret scan
    run_trivy_scan "secret" "${OUTPUT_DIR}/trivy-${SCAN_TARGET}-secret.txt" "--scanners secret"
    
    # Run SARIF scan for GitHub integration
    run_sarif_scan
    
    # Display summary
    echo -e "${GREEN}📋 Scan Summary:${NC}"
    echo -e "  - Target: ${IMAGE_NAME}:${SCAN_TARGET}"
    echo -e "  - Vulnerability report: ${OUTPUT_DIR}/trivy-${SCAN_TARGET}-vuln.txt"
    echo -e "  - Secret report: ${OUTPUT_DIR}/trivy-${SCAN_TARGET}-secret.txt"
    echo -e "  - SARIF report: ${OUTPUT_DIR}/trivy-${SCAN_TARGET}-results.sarif"
    
    # Clean up
    cleanup_docker
    
    echo -e "${GREEN}🎉 Trivy security scan completed successfully!${NC}"
}

# Handle script interruption
trap cleanup_docker EXIT

# Run main function
main "$@"
