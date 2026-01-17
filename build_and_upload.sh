#!/bin/bash
# Build and upload stem-separator to PyPI
# Usage: ./build_and_upload.sh [--test]
#   --test    Upload to TestPyPI instead of production PyPI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
USE_TEST_PYPI=false
for arg in "$@"; do
    case $arg in
        --test)
            USE_TEST_PYPI=true
            shift
            ;;
    esac
done

# Check for required tools
echo_info "Checking required tools..."

if ! command -v python3 &> /dev/null; then
    echo_error "python3 is not installed"
    exit 1
fi

# Create/activate virtual environment for build tools
VENV_DIR="$SCRIPT_DIR/.build-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo_info "Creating virtual environment for build tools..."
    python3 -m venv "$VENV_DIR"
fi

echo_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Ensure build and twine are installed
echo_info "Installing/upgrading build tools..."
pip install --upgrade pip build twine --quiet

# Clean previous builds
echo_info "Cleaning previous build artifacts..."
rm -rf dist/ build/ *.egg-info stem_separator.egg-info/

# Build the package
echo_info "Building the package..."
python3 -m build

# Verify the build
echo_info "Verifying the build..."
python3 -m twine check dist/*

# Show what will be uploaded
echo ""
echo_info "Built packages:"
ls -la dist/

# Get version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo ""
echo_info "Package version: $VERSION"

# Confirm upload
echo ""
if [ "$USE_TEST_PYPI" = true ]; then
    echo_warn "Target: TestPyPI (https://test.pypi.org)"
    REPOSITORY_URL="https://test.pypi.org/legacy/"
    TOKEN_URL="https://test.pypi.org/manage/account/token/"
else
    echo_warn "Target: Production PyPI (https://pypi.org)"
    REPOSITORY_URL=""
    TOKEN_URL="https://pypi.org/manage/account/token/"
fi

echo ""
read -p "Do you want to upload to PyPI? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo_info "Upload cancelled."
    exit 0
fi

# Get API token if not already set
if [ -z "$TWINE_PASSWORD" ]; then
    echo ""
    echo_info "API Token required."
    echo "  Get your token at: $TOKEN_URL"
    echo "  Paste the FULL token (starting with 'pypi-')"
    echo ""
    read -sp "Enter API token: " PYPI_TOKEN
    echo ""

    if [ -z "$PYPI_TOKEN" ]; then
        echo_error "No token provided. Upload cancelled."
        exit 1
    fi

    export TWINE_USERNAME="__token__"
    export TWINE_PASSWORD="$PYPI_TOKEN"
fi

# Upload to PyPI
echo ""
echo_info "Uploading to PyPI..."

if [ "$USE_TEST_PYPI" = true ]; then
    twine upload --repository-url "$REPOSITORY_URL" dist/*
else
    twine upload dist/*
fi

echo ""
echo_info "Upload complete!"

if [ "$USE_TEST_PYPI" = true ]; then
    echo_info "View your package at: https://test.pypi.org/project/stem-separator/"
    echo_info "Install with: pip install --index-url https://test.pypi.org/simple/ stem-separator"
else
    echo_info "View your package at: https://pypi.org/project/stem-separator/"
    echo_info "Install with: pip install stem-separator"
fi
