#!/bin/bash

# Docker Build Script for Sustainable Credit Risk AI
# This script handles the Docker build process with proper data file handling

set -e

# Configuration
IMAGE_NAME="sustainable-credit-risk-ai"
DOCKERFILE_PATH="infrastructure/docker/Dockerfile"
BUILD_CONTEXT="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] TARGET"
    echo ""
    echo "TARGET options:"
    echo "  production   - Build production image"
    echo "  development  - Build development image"
    echo "  training     - Build training image"
    echo "  inference    - Build inference image"
    echo ""
    echo "OPTIONS:"
    echo "  -t, --tag TAG     - Tag for the image (default: latest)"
    echo "  -p, --push        - Push image to registry after build"
    echo "  -c, --cache       - Use build cache"
    echo "  -h, --help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 production"
    echo "  $0 -t v1.0.0 -p production"
    echo "  $0 -c inference"
}

# Function to check if data file exists
check_data_file() {
    if [ ! -f "Bank_data.csv" ]; then
        echo -e "${YELLOW}⚠️  Warning: Bank_data.csv not found in current directory${NC}"
        echo -e "${YELLOW}   The build may fail if the data file is required${NC}"
        echo -e "${YELLOW}   Consider creating a sample data file or updating the Dockerfile${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Bank_data.csv found${NC}"
        return 0
    fi
}

# Function to create sample data file if missing
create_sample_data() {
    echo -e "${YELLOW}📝 Creating sample Bank_data.csv...${NC}"
    cat > Bank_data.csv << EOF
customer_id,age,gender,marital_status,employment_type,annual_income_inr,loan_amount_inr,loan_purpose,loan_term_months,credit_score,num_open_credit_accounts,num_delinquent_accounts,debt_to_income_ratio,past_default,default
SAMPLE001,35,Male,Married,Salaried,500000,100000,Personal Loan,24,750,3,0,0.3,0,0
SAMPLE002,28,Female,Single,Salaried,400000,80000,Credit Card,12,720,2,0,0.25,0,0
EOF
    echo -e "${GREEN}✅ Sample data file created${NC}"
}

# Function to build Docker image
build_image() {
    local target="$1"
    local tag="$2"
    local use_cache="$3"
    
    echo -e "${BLUE}🔨 Building Docker image: ${IMAGE_NAME}:${tag} (target: ${target})${NC}"
    
    # Build command
    local build_cmd="docker build"
    
    # Add cache options if requested
    if [ "$use_cache" = "true" ]; then
        build_cmd="$build_cmd --cache-from type=gha --cache-to type=gha,mode=max"
    fi
    
    # Add build arguments
    build_cmd="$build_cmd --file ${DOCKERFILE_PATH}"
    build_cmd="$build_cmd --target ${target}"
    build_cmd="$build_cmd --tag ${IMAGE_NAME}:${tag}"
    build_cmd="$build_cmd ${BUILD_CONTEXT}"
    
    echo -e "${YELLOW}Running: ${build_cmd}${NC}"
    
    if ! eval "$build_cmd"; then
        echo -e "${RED}❌ Docker build failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Docker image built successfully: ${IMAGE_NAME}:${tag}${NC}"
    return 0
}

# Function to push image
push_image() {
    local tag="$1"
    
    echo -e "${BLUE}📤 Pushing image: ${IMAGE_NAME}:${tag}${NC}"
    
    if ! docker push "${IMAGE_NAME}:${tag}"; then
        echo -e "${RED}❌ Failed to push image${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
    return 0
}

# Function to display image info
show_image_info() {
    local tag="$1"
    
    echo -e "${BLUE}📊 Image Information:${NC}"
    docker images "${IMAGE_NAME}:${tag}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

# Parse command line arguments
TARGET=""
TAG="latest"
PUSH=false
USE_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -c|--cache)
            USE_CACHE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        production|development|training|inference)
            TARGET="$1"
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate target
if [ -z "$TARGET" ]; then
    echo -e "${RED}❌ Error: Target is required${NC}"
    usage
    exit 1
fi

# Main execution
main() {
    echo -e "${GREEN}🚀 Starting Docker build process${NC}"
    echo -e "Target: ${TARGET}"
    echo -e "Tag: ${TAG}"
    echo -e "Push: ${PUSH}"
    echo -e "Cache: ${USE_CACHE}"
    echo ""
    
    # Check if data file exists
    if ! check_data_file; then
        echo -e "${YELLOW}Would you like to create a sample data file? (y/N)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            create_sample_data
        else
            echo -e "${YELLOW}⚠️  Proceeding without data file...${NC}"
        fi
    fi
    
    # Build the image
    if ! build_image "$TARGET" "$TAG" "$USE_CACHE"; then
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    fi
    
    # Show image info
    show_image_info "$TAG"
    
    # Push if requested
    if [ "$PUSH" = "true" ]; then
        if ! push_image "$TAG"; then
            echo -e "${RED}❌ Push failed${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}🎉 Docker build process completed successfully!${NC}"
}

# Run main function
main
