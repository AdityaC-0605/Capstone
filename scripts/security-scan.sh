#!/bin/bash
# Container Security Scanning Script for Sustainable Credit Risk AI System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="credit-risk-ai"
DOCKERFILE_PATH="."
TRIVY_CACHE_DIR="./trivy-cache"
REPORT_DIR="./security-reports"

# Create directories
mkdir -p "$TRIVY_CACHE_DIR" "$REPORT_DIR"

echo -e "${BLUE}🔒 Container Security Scanning for Credit Risk AI System${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the image if it doesn't exist
print_status "Checking if image exists..."
if ! docker image inspect "$IMAGE_NAME:latest" > /dev/null 2>&1; then
    print_status "Building Docker image..."
    docker build -t "$IMAGE_NAME:latest" "$DOCKERFILE_PATH"
    print_success "Image built successfully"
else
    print_status "Image already exists"
fi

# Install Trivy if not available
if ! command -v trivy &> /dev/null; then
    print_status "Installing Trivy..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install aquasecurity/trivy/trivy
        else
            print_error "Homebrew not found. Please install Trivy manually."
            exit 1
        fi
    else
        print_error "Unsupported OS. Please install Trivy manually."
        exit 1
    fi
    print_success "Trivy installed successfully"
fi

# Update Trivy database
print_status "Updating Trivy vulnerability database..."
trivy image --download-db-only --cache-dir "$TRIVY_CACHE_DIR"
print_success "Database updated"

# Scan for vulnerabilities
print_status "Scanning for vulnerabilities..."
VULN_REPORT="$REPORT_DIR/vulnerabilities-$(date +%Y%m%d-%H%M%S).json"
trivy image \
    --cache-dir "$TRIVY_CACHE_DIR" \
    --format json \
    --output "$VULN_REPORT" \
    "$IMAGE_NAME:latest"

# Generate human-readable report
VULN_REPORT_TXT="$REPORT_DIR/vulnerabilities-$(date +%Y%m%d-%H%M%S).txt"
trivy image \
    --cache-dir "$TRIVY_CACHE_DIR" \
    --format table \
    --output "$VULN_REPORT_TXT" \
    "$IMAGE_NAME:latest"

print_success "Vulnerability scan completed: $VULN_REPORT_TXT"

# Scan for secrets
print_status "Scanning for secrets and sensitive information..."
SECRET_REPORT="$REPORT_DIR/secrets-$(date +%Y%m%d-%H%M%S).json"
trivy image \
    --cache-dir "$TRIVY_CACHE_DIR" \
    --scanners secret \
    --format json \
    --output "$SECRET_REPORT" \
    "$IMAGE_NAME:latest"

print_success "Secret scan completed: $SECRET_REPORT"

# Scan for misconfigurations
print_status "Scanning for misconfigurations..."
CONFIG_REPORT="$REPORT_DIR/config-$(date +%Y%m%d-%H%M%S).json"
trivy image \
    --cache-dir "$TRIVY_CACHE_DIR" \
    --scanners config \
    --format json \
    --output "$CONFIG_REPORT" \
    "$IMAGE_NAME:latest"

print_success "Configuration scan completed: $CONFIG_REPORT"

# Docker Bench Security (if available)
if command -v docker-bench-security &> /dev/null; then
    print_status "Running Docker Bench Security..."
    BENCH_REPORT="$REPORT_DIR/docker-bench-$(date +%Y%m%d-%H%M%S).log"
    docker run --rm --net host --pid host --userns host --cap-add audit_control \
        -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
        -v /etc:/etc:ro \
        -v /usr/bin/containerd:/usr/bin/containerd:ro \
        -v /usr/bin/runc:/usr/bin/runc:ro \
        -v /usr/lib/systemd:/usr/lib/systemd:ro \
        -v /var/lib:/var/lib:ro \
        -v /var/run/docker.sock:/var/run/docker.sock:ro \
        --label docker_bench_security \
        docker/docker-bench-security > "$BENCH_REPORT" 2>&1
    print_success "Docker Bench Security completed: $BENCH_REPORT"
else
    print_warning "Docker Bench Security not available. Skipping..."
fi

# Analyze results
print_status "Analyzing scan results..."

# Count vulnerabilities by severity
CRITICAL=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' "$VULN_REPORT" 2>/dev/null | wc -l || echo "0")
HIGH=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH") | .VulnerabilityID' "$VULN_REPORT" 2>/dev/null | wc -l || echo "0")
MEDIUM=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity=="MEDIUM") | .VulnerabilityID' "$VULN_REPORT" 2>/dev/null | wc -l || echo "0")
LOW=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity=="LOW") | .VulnerabilityID' "$VULN_REPORT" 2>/dev/null | wc -l || echo "0")

# Count secrets
SECRETS=$(jq '.Results[]?.Secrets[]? | .RuleID' "$SECRET_REPORT" 2>/dev/null | wc -l || echo "0")

# Count misconfigurations
MISCONFIGS=$(jq '.Results[]?.Misconfigurations[]? | .ID' "$CONFIG_REPORT" 2>/dev/null | wc -l || echo "0")

echo ""
echo "📊 Security Scan Summary"
echo "========================"
echo "Vulnerabilities:"
echo "  🔴 Critical: $CRITICAL"
echo "  🟠 High:     $HIGH"
echo "  🟡 Medium:   $MEDIUM"
echo "  🟢 Low:      $LOW"
echo ""
echo "🔐 Secrets found: $SECRETS"
echo "⚙️  Misconfigurations: $MISCONFIGS"
echo ""

# Generate summary report
SUMMARY_REPORT="$REPORT_DIR/security-summary-$(date +%Y%m%d-%H%M%S).json"
cat > "$SUMMARY_REPORT" << EOF
{
  "scan_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "image": "$IMAGE_NAME:latest",
  "vulnerabilities": {
    "critical": $CRITICAL,
    "high": $HIGH,
    "medium": $MEDIUM,
    "low": $LOW,
    "total": $((CRITICAL + HIGH + MEDIUM + LOW))
  },
  "secrets": $SECRETS,
  "misconfigurations": $MISCONFIGS,
  "reports": {
    "vulnerabilities": "$VULN_REPORT",
    "secrets": "$SECRET_REPORT",
    "configurations": "$CONFIG_REPORT"
  }
}
EOF

print_success "Summary report generated: $SUMMARY_REPORT"

# Security recommendations
echo ""
echo "🛡️  Security Recommendations"
echo "============================"

if [ "$CRITICAL" -gt 0 ] || [ "$HIGH" -gt 0 ]; then
    print_error "High/Critical vulnerabilities found! Please review and update base images."
    echo "  - Update base image to latest version"
    echo "  - Review and update package dependencies"
    echo "  - Consider using distroless or minimal base images"
fi

if [ "$SECRETS" -gt 0 ]; then
    print_error "Secrets detected in image! Please remove sensitive information."
    echo "  - Use environment variables for secrets"
    echo "  - Implement proper secret management"
    echo "  - Review .dockerignore file"
fi

if [ "$MISCONFIGS" -gt 0 ]; then
    print_warning "Configuration issues found. Please review Dockerfile best practices."
    echo "  - Run containers as non-root user"
    echo "  - Use specific image tags instead of 'latest'"
    echo "  - Minimize attack surface by removing unnecessary packages"
fi

# Best practices check
echo ""
echo "✅ Security Best Practices Checklist"
echo "===================================="
echo "□ Multi-stage builds used to minimize image size"
echo "□ Non-root user configured"
echo "□ Specific base image tags used (not 'latest')"
echo "□ Minimal base images used (Alpine, distroless)"
echo "□ Unnecessary packages removed"
echo "□ Health checks implemented"
echo "□ Security headers configured in web server"
echo "□ Secrets managed externally (not in image)"
echo "□ Regular security scans in CI/CD pipeline"
echo "□ Image signing and verification enabled"

# Exit with appropriate code
if [ "$CRITICAL" -gt 0 ]; then
    print_error "Critical vulnerabilities found. Failing build."
    exit 1
elif [ "$HIGH" -gt 5 ]; then
    print_warning "Too many high-severity vulnerabilities found."
    exit 1
elif [ "$SECRETS" -gt 0 ]; then
    print_error "Secrets found in image. Failing build."
    exit 1
else
    print_success "Security scan completed successfully!"
    exit 0
fi