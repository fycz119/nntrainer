#!/bin/bash

# Clean, rebuild, and install CausalLM Android artifacts.
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_DIR="/data/local/tmp/nntrainer/causallm"

log_header "Clean Build and Install CausalLM Android"
log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
log_info "SCRIPT_DIR: $SCRIPT_DIR"
log_info "INSTALL_DIR: $INSTALL_DIR"

log_step "1/5" "Clean local build artifacts"
rm -rf "$SCRIPT_DIR/jni/libs" "$SCRIPT_DIR/jni/obj" "$NNTRAINER_ROOT/builddir"
log_success "Local build artifacts removed"

log_step "2/5" "Clean device install artifacts"
adb shell "rm -rf $INSTALL_DIR/lib* $INSTALL_DIR/nnt* $INSTALL_DIR/run*"
log_success "Device artifacts removed"

log_step "3/5" "Build CausalLM Android application"
"$SCRIPT_DIR/build_android.sh"

log_step "4/5" "Build CausalLM API library"
"$SCRIPT_DIR/build_api_lib.sh"

log_step "5/5" "Install CausalLM to Android device"
"$SCRIPT_DIR/install_android.sh"

log_header "Done"
log_success "CausalLM Android artifacts rebuilt and installed"
