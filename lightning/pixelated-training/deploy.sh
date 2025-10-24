#!/bin/bash
# Production deployment script for Wayfarer model infrastructure
# Handles environment setup, model deployment, and health verification

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="wayfarer-production"
ENVIRONMENT="${ENVIRONMENT:-production}"
GPU_COUNT="${GPU_COUNT:-3}"
CHECKPOINT_PRIMARY="${CHECKPOINT_PRIMARY:-checkpoint-300}"
CHECKPOINT_FALLBACK="${CHECKPOINT_FALLBACK:-checkpoint-200}"
CHECKPOINT_CRISIS="${CHECKPOINT_CRISIS:-checkpoint-350}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose not found. Please install Docker Compose."
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        warn "NVIDIA Docker runtime not detected. GPU support may not work."
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        log "✅ Found $GPU_AVAILABLE GPU(s)"
        
        if [ "$GPU_AVAILABLE" -lt "$GPU_COUNT" ]; then
            warn "Requested $GPU_COUNT GPUs but only $GPU_AVAILABLE available"
            GPU_COUNT=$GPU_AVAILABLE
        fi
    else
        warn "nvidia-smi not found. Cannot verify GPU availability."
    fi
    
    # Check disk space (need at least 50GB for models and logs)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=52428800  # 50GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        error "Insufficient disk space. Need at least 50GB, have $(($AVAILABLE_SPACE / 1048576))GB"
    fi
    
    log "✅ Prerequisites check passed"
}

# Validate model checkpoints
validate_checkpoints() {
    log "🔍 Validating model checkpoints..."
    
    for checkpoint in "$CHECKPOINT_PRIMARY" "$CHECKPOINT_FALLBACK" "$CHECKPOINT_CRISIS"; do
        checkpoint_path="wayfarer-balanced/$checkpoint"
        
        if [ ! -d "$checkpoint_path" ]; then
            error "Checkpoint not found: $checkpoint_path"
        fi
        
        # Check for required files
        required_files=("adapter_config.json" "adapter_model.safetensors")
        for file in "${required_files[@]}"; do
            if [ ! -f "$checkpoint_path/$file" ]; then
                error "Required file missing: $checkpoint_path/$file"
            fi
        done
        
        log "✅ Validated checkpoint: $checkpoint"
    done
}

# Setup environment
setup_environment() {
    log "🏗️ Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs model_cache monitoring/logs ssl
    
    # Generate environment file
    cat > .env << EOF
# Wayfarer Production Environment
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=1.0.0
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-wayfarer2024}
MODEL_CACHE_PATH=$(pwd)/model_cache
ENVIRONMENT=$ENVIRONMENT
GPU_COUNT=$GPU_COUNT
CHECKPOINT_PRIMARY=$CHECKPOINT_PRIMARY
CHECKPOINT_FALLBACK=$CHECKPOINT_FALLBACK
CHECKPOINT_CRISIS=$CHECKPOINT_CRISIS
EOF
    
    # Set proper permissions
    chmod 755 logs model_cache
    chmod 600 .env
    
    log "✅ Environment setup complete"
}

# Build Docker images
build_images() {
    log "🐳 Building Docker images..."
    
    # Build production image
    docker build \
        -f Dockerfile.production \
        -t wayfarer-api:latest \
        -t wayfarer-api:$(git rev-parse --short HEAD 2>/dev/null || echo "latest") \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="1.0.0" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")" \
        .
    
    # Build autoscaler image
    docker build \
        -f autoscaler/Dockerfile \
        -t wayfarer-autoscaler:latest \
        autoscaler/
    
    log "✅ Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "🚀 Deploying services..."
    
    # Stop any existing services
    docker-compose -f docker-compose.production.yml down --remove-orphans
    
    # Start core services (without optional components)
    docker-compose -f docker-compose.production.yml up -d \
        wayfarer-lb \
        wayfarer-primary \
        wayfarer-fallback \
        wayfarer-cache \
        prometheus \
        grafana \
        node-exporter \
        gpu-exporter \
        fluent-bit
    
    # Wait for core services to be healthy
    log "⏳ Waiting for services to become healthy..."
    
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose -f docker-compose.production.yml ps | grep -q "unhealthy\|starting"; then
            log "Services still starting... (attempt $((attempt+1))/$max_attempts)"
            sleep 10
            ((attempt++))
        else
            break
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error "Services failed to become healthy within timeout"
    fi
    
    log "✅ Core services deployed successfully"
}

# Deploy optional services
deploy_optional_services() {
    log "🔧 Deploying optional services..."
    
    # Deploy crisis model if requested
    if [ "${DEPLOY_CRISIS:-false}" = "true" ]; then
        log "Deploying crisis model..."
        docker-compose -f docker-compose.production.yml --profile crisis-enabled up -d wayfarer-crisis
    fi
    
    # Deploy autoscaler if requested
    if [ "${DEPLOY_AUTOSCALER:-false}" = "true" ]; then
        log "Deploying autoscaler..."
        docker-compose -f docker-compose.production.yml --profile autoscaling up -d autoscaler
    fi
    
    log "✅ Optional services deployed"
}

# Verify deployment
verify_deployment() {
    log "🔍 Verifying deployment..."
    
    # Check service health
    services=("wayfarer-primary" "wayfarer-fallback" "wayfarer-cache" "prometheus" "grafana")
    
    for service in "${services[@]}"; do
        if ! docker-compose -f docker-compose.production.yml ps | grep -q "$service.*Up"; then
            error "Service $service is not running"
        fi
        log "✅ $service is running"
    done
    
    # Test API endpoints
    log "Testing API endpoints..."
    
    max_attempts=10
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f http://localhost:8000/health > /dev/null; then
            log "✅ API health check passed"
            break
        else
            log "API not ready... (attempt $((attempt+1))/$max_attempts)"
            sleep 5
            ((attempt++))
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error "API health check failed"
    fi
    
    # Test model inference
    log "Testing model inference..."
    
    response=$(curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello, this is a test", "max_length": 50, "temperature": 0.7}')
    
    if echo "$response" | grep -q "response"; then
        log "✅ Model inference test passed"
    else
        error "Model inference test failed: $response"
    fi
    
    # Display service URLs
    log "📊 Service URLs:"
    echo "  API: http://localhost:8000"
    echo "  Health: http://localhost:8000/health"
    echo "  Grafana: http://localhost:3000 (admin/wayfarer2024)"
    echo "  Prometheus: http://localhost:9090"
    
    log "✅ Deployment verification complete"
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up..."
    
    # Remove temporary files
    rm -f .env.tmp
    
    # Prune unused Docker resources
    docker system prune -f
    
    log "✅ Cleanup complete"
}

# Main deployment flow
main() {
    log "🚀 Starting Wayfarer production deployment"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run deployment steps
    check_prerequisites
    validate_checkpoints
    setup_environment
    build_images
    deploy_services
    deploy_optional_services
    verify_deployment
    
    log "🎉 Wayfarer production deployment completed successfully!"
    log "🔗 Access your deployment at http://localhost:8000"
    log "📊 Monitor performance at http://localhost:3000"
    
    # Display next steps
    echo ""
    echo "Next steps:"
    echo "1. Run A/B testing: ./test-checkpoints.sh"
    echo "2. Monitor performance in Grafana"
    echo "3. Check logs: docker-compose -f docker-compose.production.yml logs -f"
    echo "4. Scale services: DEPLOY_AUTOSCALER=true ./deploy.sh"
}

# Handle script termination
trap cleanup EXIT

# Run deployment
main "$@"