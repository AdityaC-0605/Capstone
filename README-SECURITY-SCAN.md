# Security Scanning with Trivy

This document provides solutions for the Trivy security scanning issues encountered in the CI/CD pipeline.

## Problem Analysis

The original Trivy scan was failing with a "no space left on device" error due to:

1. **Large Docker Image**: Multi-stage Dockerfile creates substantial images
2. **Disk Space Constraints**: GitHub Actions runners have limited disk space
3. **Inefficient Scanning**: Full vulnerability database download and image extraction
4. **No Cleanup**: Docker resources not cleaned up after scanning

## Solutions Implemented

### 1. Optimized GitHub Actions Workflows

#### Basic Security Scan (`.github/workflows/security-scan.yml`)
- Uses optimized Trivy configuration
- Implements proper cleanup procedures
- Focuses on critical and high severity vulnerabilities
- Uses caching to reduce download time

#### Advanced Security Scan (`.github/workflows/optimized-security-scan.yml`)
- Two-stage scanning approach:
  - Lightweight scan for all branches
  - Full production scan only for main branch
- Uses inference target for faster scanning
- Implements comprehensive cleanup

### 2. Enhanced Docker Configuration

#### Optimized Dockerfile
- Added better cleanup in inference stage
- Removed unnecessary files and caches
- Optimized layer structure

#### Docker Ignore File (`.dockerignore`)
- Excludes unnecessary files from Docker context
- Reduces image size and build time
- Improves scanning efficiency

### 3. Local Scanning Script

#### Trivy Scan Script (`scripts/trivy-scan.sh`)
- Comprehensive local scanning solution
- Disk space monitoring
- Automatic cleanup
- Multiple output formats (table, SARIF)
- Color-coded output for better readability

## Usage Instructions

### GitHub Actions

The workflows will automatically run on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### Local Scanning

```bash
# Scan production image
./scripts/trivy-scan.sh production

# Scan inference image (lighter)
./scripts/trivy-scan.sh inference

# Scan development image
./scripts/trivy-scan.sh development
```

### Manual Trivy Commands

```bash
# Quick vulnerability scan
trivy image --severity CRITICAL,HIGH sustainable-credit-risk-ai:production

# Full scan with SARIF output
trivy image --format sarif --output trivy-results.sarif sustainable-credit-risk-ai:production

# Secret scanning only
trivy image --scanners secret sustainable-credit-risk-ai:production
```

## Optimization Features

### 1. Disk Space Management
- Pre-scan disk space checks
- Automatic Docker cleanup
- Efficient caching strategies

### 2. Scanning Optimizations
- Skip version checks to avoid warnings
- Ignore unfixed vulnerabilities
- Focus on critical and high severity issues
- Use external cache directories

### 3. Resource Cleanup
- Automatic Docker system pruning
- Volume cleanup
- Builder cache cleanup
- Temporary file removal

## Configuration Options

### Trivy Configuration
```yaml
scanners: 'vuln,secret'          # Scan types
severity: 'CRITICAL,HIGH,MEDIUM' # Severity levels
skip-version-check: true         # Skip version warnings
ignore-unfixed: true            # Ignore unfixed vulnerabilities
cache-dir: '.trivycache'        # Cache directory
```

### Docker Optimizations
```dockerfile
# Environment variables for optimization
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Cleanup commands
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y
```

## Troubleshooting

### Common Issues

1. **Disk Space Errors**
   - Use the lightweight inference image for scanning
   - Ensure proper cleanup procedures
   - Monitor disk space before scanning

2. **Slow Scanning**
   - Use caching with `--cache-dir`
   - Limit severity levels
   - Use `--ignore-unfixed` flag

3. **False Positives**
   - Review and whitelist known false positives
   - Use `--ignore-unfixed` for development scans
   - Focus on critical and high severity issues

### Performance Tips

1. **Use Smaller Images**: Scan the inference target instead of full production
2. **Enable Caching**: Use persistent cache directories
3. **Limit Scope**: Focus on critical vulnerabilities for faster scans
4. **Cleanup Regularly**: Implement proper cleanup procedures

## Security Best Practices

1. **Regular Scanning**: Run scans on every commit and PR
2. **Severity Focus**: Prioritize critical and high severity issues
3. **Automated Remediation**: Integrate with CI/CD for automatic blocking
4. **Documentation**: Maintain security reports and remediation plans

## Integration with CI/CD

The security scans are integrated into the GitLab CI pipeline:

```yaml
security-scan:
  stage: security
  script:
    - docker build -t $IMAGE_NAME:$CI_COMMIT_SHA .
    - trivy image --format sarif --output trivy-results.sarif $IMAGE_NAME:$CI_COMMIT_SHA
  artifacts:
    reports:
      sast: trivy-results.sarif
```

## Monitoring and Alerting

- SARIF reports are uploaded to GitHub Security tab
- Failed scans block deployments
- Security reports are archived for compliance
- Integration with monitoring systems for alerting

## Compliance and Reporting

- SARIF format for standard security reporting
- Integration with security dashboards
- Automated compliance checks
- Audit trail maintenance
