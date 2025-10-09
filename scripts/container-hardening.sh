#!/bin/bash
# Container Hardening Script for Sustainable Credit Risk AI System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛡️  Container Hardening for Credit Risk AI System${NC}"
echo "================================================="

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

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. Some hardening steps may not apply."
fi

print_status "Applying container hardening measures..."

# 1. Docker daemon hardening
print_status "Checking Docker daemon configuration..."

DOCKER_CONFIG_DIR="/etc/docker"
DOCKER_CONFIG_FILE="$DOCKER_CONFIG_DIR/daemon.json"

if [ -f "$DOCKER_CONFIG_FILE" ]; then
    print_status "Docker daemon configuration exists"
else
    print_status "Creating Docker daemon configuration..."
    sudo mkdir -p "$DOCKER_CONFIG_DIR"
    
    # Create hardened Docker daemon configuration
    sudo tee "$DOCKER_CONFIG_FILE" > /dev/null << 'EOF'
{
  "icc": false,
  "userns-remap": "default",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp.json",
  "apparmor-profile": "docker-default",
  "selinux-enabled": true,
  "disable-legacy-registry": true,
  "experimental": false,
  "metrics-addr": "127.0.0.1:9323",
  "tls": true,
  "tlsverify": true,
  "tlscert": "/etc/docker/certs/server-cert.pem",
  "tlskey": "/etc/docker/certs/server-key.pem",
  "tlscacert": "/etc/docker/certs/ca.pem"
}
EOF
    print_success "Docker daemon configuration created"
fi

# 2. Create custom seccomp profile
print_status "Creating custom seccomp profile..."
sudo tee "/etc/docker/seccomp.json" > /dev/null << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "archMap": [
    {
      "architecture": "SCMP_ARCH_X86_64",
      "subArchitectures": [
        "SCMP_ARCH_X86",
        "SCMP_ARCH_X32"
      ]
    }
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "access",
        "adjtimex",
        "alarm",
        "bind",
        "brk",
        "capget",
        "capset",
        "chdir",
        "chmod",
        "chown",
        "chown32",
        "clock_getres",
        "clock_gettime",
        "clock_nanosleep",
        "close",
        "connect",
        "copy_file_range",
        "creat",
        "dup",
        "dup2",
        "dup3",
        "epoll_create",
        "epoll_create1",
        "epoll_ctl",
        "epoll_pwait",
        "epoll_wait",
        "eventfd",
        "eventfd2",
        "execve",
        "execveat",
        "exit",
        "exit_group",
        "faccessat",
        "fadvise64",
        "fadvise64_64",
        "fallocate",
        "fanotify_mark",
        "fchdir",
        "fchmod",
        "fchmodat",
        "fchown",
        "fchown32",
        "fchownat",
        "fcntl",
        "fcntl64",
        "fdatasync",
        "fgetxattr",
        "flistxattr",
        "flock",
        "fork",
        "fremovexattr",
        "fsetxattr",
        "fstat",
        "fstat64",
        "fstatat64",
        "fstatfs",
        "fstatfs64",
        "fsync",
        "ftruncate",
        "ftruncate64",
        "futex",
        "getcwd",
        "getdents",
        "getdents64",
        "getegid",
        "getegid32",
        "geteuid",
        "geteuid32",
        "getgid",
        "getgid32",
        "getgroups",
        "getgroups32",
        "getitimer",
        "getpeername",
        "getpgid",
        "getpgrp",
        "getpid",
        "getppid",
        "getpriority",
        "getrandom",
        "getresgid",
        "getresgid32",
        "getresuid",
        "getresuid32",
        "getrlimit",
        "get_robust_list",
        "getrusage",
        "getsid",
        "getsockname",
        "getsockopt",
        "get_thread_area",
        "gettid",
        "gettimeofday",
        "getuid",
        "getuid32",
        "getxattr",
        "inotify_add_watch",
        "inotify_init",
        "inotify_init1",
        "inotify_rm_watch",
        "io_cancel",
        "ioctl",
        "io_destroy",
        "io_getevents",
        "ioprio_get",
        "ioprio_set",
        "io_setup",
        "io_submit",
        "ipc",
        "kill",
        "lchown",
        "lchown32",
        "lgetxattr",
        "link",
        "linkat",
        "listen",
        "listxattr",
        "llistxattr",
        "lremovexattr",
        "lseek",
        "lsetxattr",
        "lstat",
        "lstat64",
        "madvise",
        "memfd_create",
        "mincore",
        "mkdir",
        "mkdirat",
        "mknod",
        "mknodat",
        "mlock",
        "mlock2",
        "mlockall",
        "mmap",
        "mmap2",
        "mprotect",
        "mq_getsetattr",
        "mq_notify",
        "mq_open",
        "mq_timedreceive",
        "mq_timedsend",
        "mq_unlink",
        "mremap",
        "msgctl",
        "msgget",
        "msgrcv",
        "msgsnd",
        "msync",
        "munlock",
        "munlockall",
        "munmap",
        "nanosleep",
        "newfstatat",
        "_newselect",
        "open",
        "openat",
        "pause",
        "pipe",
        "pipe2",
        "poll",
        "ppoll",
        "prctl",
        "pread64",
        "preadv",
        "prlimit64",
        "pselect6",
        "ptrace",
        "pwrite64",
        "pwritev",
        "read",
        "readahead",
        "readlink",
        "readlinkat",
        "readv",
        "recv",
        "recvfrom",
        "recvmmsg",
        "recvmsg",
        "remap_file_pages",
        "removexattr",
        "rename",
        "renameat",
        "renameat2",
        "restart_syscall",
        "rmdir",
        "rt_sigaction",
        "rt_sigpending",
        "rt_sigprocmask",
        "rt_sigqueueinfo",
        "rt_sigreturn",
        "rt_sigsuspend",
        "rt_sigtimedwait",
        "rt_tgsigqueueinfo",
        "sched_getaffinity",
        "sched_getattr",
        "sched_getparam",
        "sched_get_priority_max",
        "sched_get_priority_min",
        "sched_getscheduler",
        "sched_setaffinity",
        "sched_setattr",
        "sched_setparam",
        "sched_setscheduler",
        "sched_yield",
        "seccomp",
        "select",
        "semctl",
        "semget",
        "semop",
        "semtimedop",
        "send",
        "sendfile",
        "sendfile64",
        "sendmmsg",
        "sendmsg",
        "sendto",
        "setfsgid",
        "setfsgid32",
        "setfsuid",
        "setfsuid32",
        "setgid",
        "setgid32",
        "setgroups",
        "setgroups32",
        "setitimer",
        "setpgid",
        "setpriority",
        "setregid",
        "setregid32",
        "setresgid",
        "setresgid32",
        "setresuid",
        "setresuid32",
        "setreuid",
        "setreuid32",
        "setrlimit",
        "set_robust_list",
        "setsid",
        "setsockopt",
        "set_thread_area",
        "set_tid_address",
        "setuid",
        "setuid32",
        "setxattr",
        "shmat",
        "shmctl",
        "shmdt",
        "shmget",
        "shutdown",
        "sigaltstack",
        "signalfd",
        "signalfd4",
        "sigreturn",
        "socket",
        "socketcall",
        "socketpair",
        "splice",
        "stat",
        "stat64",
        "statfs",
        "statfs64",
        "statx",
        "symlink",
        "symlinkat",
        "sync",
        "sync_file_range",
        "syncfs",
        "sysinfo",
        "tee",
        "tgkill",
        "time",
        "timer_create",
        "timer_delete",
        "timerfd_create",
        "timerfd_gettime",
        "timerfd_settime",
        "timer_getoverrun",
        "timer_gettime",
        "timer_settime",
        "times",
        "tkill",
        "truncate",
        "truncate64",
        "ugetrlimit",
        "umask",
        "uname",
        "unlink",
        "unlinkat",
        "utime",
        "utimensat",
        "utimes",
        "vfork",
        "vmsplice",
        "wait4",
        "waitid",
        "waitpid",
        "write",
        "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF
print_success "Custom seccomp profile created"

# 3. Create AppArmor profile
print_status "Creating AppArmor profile..."
sudo tee "/etc/apparmor.d/docker-credit-risk-ai" > /dev/null << 'EOF'
#include <tunables/global>

profile docker-credit-risk-ai flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
  deny capability sys_ptrace,
  deny capability dac_override,
  deny capability dac_read_search,
  deny capability fowner,
  deny capability fsetid,
  deny capability kill,
  deny capability setgid,
  deny capability setuid,
  deny capability setpcap,
  deny capability linux_immutable,
  deny capability net_bind_service,
  deny capability net_broadcast,
  deny capability net_admin,
  deny capability net_raw,
  deny capability ipc_lock,
  deny capability ipc_owner,
  deny capability sys_chroot,
  deny capability sys_tty_config,
  deny capability mknod,
  deny capability lease,
  deny capability audit_write,
  deny capability audit_control,
  deny capability setfcap,
  deny capability mac_override,
  deny capability mac_admin,
  deny capability syslog,
  deny capability wake_alarm,
  deny capability block_suspend,
  
  # Allow necessary file access
  /app/** r,
  /app/src/** r,
  /app/models/** r,
  /app/logs/** rw,
  /app/compliance_reports/** rw,
  
  # System files
  /etc/passwd r,
  /etc/group r,
  /etc/nsswitch.conf r,
  /etc/hosts r,
  /etc/resolv.conf r,
  /etc/ssl/certs/** r,
  
  # Temporary files
  /tmp/** rw,
  /var/tmp/** rw,
  
  # Proc and sys (limited)
  @{PROC}/sys/kernel/random/uuid r,
  @{PROC}/cpuinfo r,
  @{PROC}/meminfo r,
  @{PROC}/stat r,
  @{PROC}/uptime r,
  @{PROC}/version r,
  
  # Network
  network inet stream,
  network inet dgram,
  network inet6 stream,
  network inet6 dgram,
  
  # Deny dangerous paths
  deny /boot/** rwklx,
  deny /dev/mem rwklx,
  deny /dev/kmem rwklx,
  deny /dev/port rwklx,
  deny /etc/shadow rwklx,
  deny /etc/gshadow rwklx,
  deny /root/** rwklx,
  deny /sys/kernel/debug/** rwklx,
  deny /sys/kernel/security/** rwklx,
  deny /sys/power/** rwklx,
  deny /proc/kcore rwklx,
  deny /proc/kallsyms rwklx,
  deny /proc/kmem rwklx,
  deny /proc/mem rwklx,
  deny /proc/sysrq-trigger rwklx,
  
  # Python specific
  /usr/bin/python3* ix,
  /usr/lib/python3*/** r,
  /usr/local/lib/python3*/** r,
}
EOF

# Load AppArmor profile
if command -v apparmor_parser &> /dev/null; then
    sudo apparmor_parser -r /etc/apparmor.d/docker-credit-risk-ai
    print_success "AppArmor profile loaded"
else
    print_warning "AppArmor not available on this system"
fi

# 4. Create Docker Compose security override
print_status "Creating security-hardened Docker Compose override..."
tee "docker-compose.security.yml" > /dev/null << 'EOF'
# Security hardening override for Docker Compose
version: '3.8'

services:
  api:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-credit-risk-ai
      - seccomp:/etc/docker/seccomp.json
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/tmp:noexec,nosuid,size=50m
    ulimits:
      nproc: 65535
      nofile:
        soft: 65535
        hard: 65535
    sysctls:
      - net.core.somaxconn=1024
    pids_limit: 100
    mem_limit: 4g
    cpus: 2.0

  training:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-credit-risk-ai
      - seccomp:/etc/docker/seccomp.json
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=500m
      - /var/tmp:noexec,nosuid,size=200m
    pids_limit: 200
    mem_limit: 8g
    cpus: 4.0

  postgres:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - FOWNER
      - SETGID
      - SETUID
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/run/postgresql:noexec,nosuid,size=100m
    pids_limit: 100
    mem_limit: 2g

  redis:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=50m
    pids_limit: 50
    mem_limit: 1g

  nginx:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /var/cache/nginx:noexec,nosuid,size=100m
      - /var/run:noexec,nosuid,size=50m
      - /tmp:noexec,nosuid,size=50m
    pids_limit: 50
    mem_limit: 512m
EOF

print_success "Security override created: docker-compose.security.yml"

# 5. Create runtime security script
print_status "Creating runtime security monitoring script..."
tee "scripts/runtime-security.sh" > /dev/null << 'EOF'
#!/bin/bash
# Runtime security monitoring for containers

# Monitor for privilege escalation attempts
docker events --filter event=exec --format "{{.Time}} {{.Actor.Attributes.name}} {{.Actor.Attributes.execID}}" &

# Monitor for suspicious network connections
netstat -tuln | grep -E ':(22|23|135|139|445|1433|3389|5432|6379)' || true

# Check for running containers with privileged mode
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --filter "label=privileged=true"

# Monitor resource usage
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
EOF

chmod +x scripts/runtime-security.sh
print_success "Runtime security monitoring script created"

# 6. Create security validation script
print_status "Creating security validation script..."
tee "scripts/validate-security.sh" > /dev/null << 'EOF'
#!/bin/bash
# Validate container security configuration

echo "🔍 Validating container security configuration..."

# Check if containers are running as non-root
echo "Checking user configuration..."
docker-compose exec api whoami | grep -v root || echo "✅ API running as non-root"
docker-compose exec training whoami | grep -v root || echo "✅ Training running as non-root"

# Check capabilities
echo "Checking capabilities..."
docker inspect $(docker-compose ps -q api) | jq '.[].HostConfig.CapDrop'
docker inspect $(docker-compose ps -q api) | jq '.[].HostConfig.CapAdd'

# Check security options
echo "Checking security options..."
docker inspect $(docker-compose ps -q api) | jq '.[].HostConfig.SecurityOpt'

# Check read-only filesystem
echo "Checking filesystem permissions..."
docker inspect $(docker-compose ps -q api) | jq '.[].HostConfig.ReadonlyRootfs'

# Check resource limits
echo "Checking resource limits..."
docker inspect $(docker-compose ps -q api) | jq '.[].HostConfig.Memory'
docker inspect $(docker-compose ps -q api) | jq '.[].HostConfig.NanoCpus'

echo "✅ Security validation completed"
EOF

chmod +x scripts/validate-security.sh
print_success "Security validation script created"

# 7. Update .env.example with security settings
print_status "Creating security environment template..."
tee ".env.security" > /dev/null << 'EOF'
# Security Configuration for Credit Risk AI System

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Database Security
DATABASE_SSL_MODE=require
DATABASE_SSL_CERT=/etc/ssl/certs/client-cert.pem
DATABASE_SSL_KEY=/etc/ssl/private/client-key.pem
DATABASE_SSL_CA=/etc/ssl/certs/ca-cert.pem

# Redis Security
REDIS_PASSWORD=your-redis-password-change-this
REDIS_SSL=true

# API Security
API_RATE_LIMIT=100
API_RATE_LIMIT_WINDOW=60
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com

# Encryption
ENCRYPTION_KEY=your-32-byte-encryption-key-here
FERNET_KEY=your-fernet-key-for-field-encryption

# Monitoring
ENABLE_METRICS=true
METRICS_AUTH_TOKEN=your-metrics-auth-token

# Logging
LOG_LEVEL=INFO
AUDIT_LOG_ENABLED=true
SECURITY_LOG_ENABLED=true

# Container Security
DOCKER_CONTENT_TRUST=1
DOCKER_BUILDKIT=1
EOF

print_success "Security environment template created"

# 8. Create security checklist
print_status "Creating security checklist..."
tee "SECURITY_CHECKLIST.md" > /dev/null << 'EOF'
# Container Security Checklist

## Pre-deployment Security Checklist

### Image Security
- [ ] Base images are from trusted sources
- [ ] Images are regularly updated
- [ ] Vulnerability scanning is performed
- [ ] No secrets in images
- [ ] Multi-stage builds used
- [ ] Minimal base images (Alpine, distroless)
- [ ] Image signing enabled

### Container Configuration
- [ ] Containers run as non-root user
- [ ] Read-only root filesystem
- [ ] No privileged containers
- [ ] Capabilities dropped (ALL) and only necessary ones added
- [ ] Security profiles applied (AppArmor/SELinux)
- [ ] Seccomp profiles configured
- [ ] Resource limits set
- [ ] PID limits configured
- [ ] No new privileges flag set

### Network Security
- [ ] Network segmentation implemented
- [ ] TLS encryption for all communications
- [ ] Firewall rules configured
- [ ] No unnecessary ports exposed
- [ ] Service mesh security (if applicable)

### Data Security
- [ ] Secrets managed externally (not in containers)
- [ ] Data encryption at rest
- [ ] Data encryption in transit
- [ ] Secure backup procedures
- [ ] Data retention policies

### Monitoring and Logging
- [ ] Security monitoring enabled
- [ ] Audit logging configured
- [ ] Intrusion detection system
- [ ] Log aggregation and analysis
- [ ] Alerting for security events

### Access Control
- [ ] RBAC implemented
- [ ] Multi-factor authentication
- [ ] Regular access reviews
- [ ] Principle of least privilege
- [ ] API authentication and authorization

### Compliance
- [ ] Regulatory compliance validated
- [ ] Security policies documented
- [ ] Incident response plan
- [ ] Regular security assessments
- [ ] Compliance reporting

## Runtime Security Monitoring

### Continuous Monitoring
- [ ] Container behavior monitoring
- [ ] Network traffic analysis
- [ ] Resource usage monitoring
- [ ] Security event correlation
- [ ] Threat intelligence integration

### Incident Response
- [ ] Incident response procedures
- [ ] Automated threat response
- [ ] Forensic capabilities
- [ ] Recovery procedures
- [ ] Post-incident analysis

## Security Maintenance

### Regular Tasks
- [ ] Security patch management
- [ ] Vulnerability assessments
- [ ] Penetration testing
- [ ] Security training
- [ ] Policy updates

### Automation
- [ ] Automated security scanning
- [ ] Continuous compliance monitoring
- [ ] Automated patch deployment
- [ ] Security orchestration
- [ ] Threat hunting automation
EOF

print_success "Security checklist created: SECURITY_CHECKLIST.md"

echo ""
echo "🎉 Container hardening completed!"
echo ""
echo "Next steps:"
echo "1. Review and customize the security configurations"
echo "2. Test the hardened containers: docker-compose -f docker-compose.yml -f docker-compose.security.yml up"
echo "3. Run security validation: ./scripts/validate-security.sh"
echo "4. Perform security scanning: ./scripts/security-scan.sh"
echo "5. Review the security checklist: SECURITY_CHECKLIST.md"
echo ""
print_success "Container hardening setup complete!"