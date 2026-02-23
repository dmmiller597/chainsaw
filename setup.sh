#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STRIDE_DIR="${SCRIPT_DIR}/stride"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_build_tools() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y build-essential
        elif command_exists yum; then
            sudo yum groupinstall -y "Development Tools"
        else
            echo "Error: Unsupported package manager. Please install build tools manually."
            return 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command_exists gcc || ! command_exists make; then
            echo "Installing Xcode Command Line Tools..."
            xcode-select --install
            read -p "Press enter after Xcode Command Line Tools installation is complete"
        fi
    else
        echo "Unsupported operating system"; return 1
    fi
}

compile_stride() {
    echo "Starting stride compilation..."
    [ -f "$STRIDE_DIR/stride.tgz" ] || { echo "Error: stride.tgz not found"; return 1; }
    tar -zxf "$STRIDE_DIR/stride.tgz" -C "$STRIDE_DIR" || return 1
    install_build_tools || return 1
    cd "$STRIDE_DIR" && make && chmod +x stride || return 1
    ./stride --help || return 1
    cd - || return 1
    echo "Stride compilation successful!"
}

# Python dependencies — install with: uv sync --extra dev
# (uv handles venv creation automatically)

if compile_stride; then
    echo "Chainsaw setup complete."
    echo "Activate your environment: source .venv/bin/activate"
    echo "Test: python get_predictions.py --structure_file example_files/AF-A0A1W2PQ64-F1-model_v4.pdb --output results/test.tsv"
else
    echo "Error during stride compilation. To compile manually:"
    echo "  cd $STRIDE_DIR && tar -zxf stride.tgz && make"
fi
