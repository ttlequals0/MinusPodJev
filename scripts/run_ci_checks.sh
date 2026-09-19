#!/bin/bash
set -e

# Run CI checks locally - mirrors .github/workflows/main.yml with uv.

RED='\033[0;31'
GREEN='\033[0;32'
YELLOW='\033[1;33'
BLUE='\033[0;34'
NC='\033[0m'

EXIT_CODE=0
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_section() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${BLUE}===========================================${NC}\n"
}

print_success() { echo -e "${GREEN}PASS: $1${NC}"; }
print_error() { echo -e "${RED}FAIL: $1${NC}"; }
print_warning() { echo -e "${YELLOW}WARN: $1${NC}"; }

run_check() {
    local name=$1
    local command=$2
    echo -e "${YELLOW}Running: $name${NC}"
    if eval "$command"; then
        print_success "$name passed"
    else
        print_error "$name failed"
        EXIT_CODE=1
    fi
}

AUTO_FIX=true
SKIP_DOCKER=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-auto-fix) AUTO_FIX=false; shift ;;
    --include-docker) SKIP_DOCKER=false; shift ;;
    -h|--help) echo "Usage: $0 [--no-auto-fix] [--include-docker]"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd "$PROJECT_ROOT"

print_section "Checking Prerequisites"
command -v python3 >/dev/null 2>&1 || { print_error "Python 3 is not installed"; exit 1; }
command -v uv >/dev/null 2>&1 || { print_error "uv is not installed"; exit 1; }
print_success "uv $(uv --version | awk '{print $2}') found"

print_section "Backend Environment"
run_check "uv sync" "uv sync"

print_section "Backend Tests"
print_success "Using SQLite for tests"
run_check "Backend tests with coverage" "TESTING=true uv run pytest -q --cov=backend/app --cov-report=term"

print_section "Python Linting"
if ! run_check "Ruff" "uv run ruff check backend/"; then
    if [ "$AUTO_FIX" = true ]; then
        echo -e "${YELLOW}Auto-fixing Ruff issues...${NC}"
        uv run ruff check backend/ --fix || true
    fi
fi
run_check "MyPy" "uv run mypy backend/app/ --ignore-missing-imports"

print_section "Frontend Checks"
if command -v node >/dev/null 2>&1; then
    cd "$PROJECT_ROOT/frontend"
    [ -d "node_modules" ] || npm ci
    run_check "Frontend TypeScript check" "npm run type-check"
    run_check "Frontend lint" "npm run lint"
    run_check "Frontend build" "npm run build"
    cd "$PROJECT_ROOT"
else
    print_warning "Node.js not installed. Skipping frontend checks."
fi

print_section "Docker Build Test"
if [ "$SKIP_DOCKER" = false ] && command -v docker >/dev/null 2>&1; then
    run_check "Docker build" "docker build -f Dockerfile -t jev-proxy:ci-test ."
    docker rmi jev-proxy:ci-test 2>/dev/null || true
elif [ "$SKIP_DOCKER" = false ]; then
    print_warning "Docker not installed. Skipping Docker build test."
fi

print_section "CI Check Summary"
if [ $EXIT_CODE -eq 0 ]; then
    print_success "All CI checks passed!"
else
    print_error "Some CI checks failed. Please fix the issues above."
fi

exit $EXIT_CODE
