#!/bin/bash

REPO_NAME="peak-flops"

echo "Creating repository structure under: $REPO_NAME"

# Core directories
mkdir -p notes/{cuda,linux,gpu-architecture}
mkdir -p experiments/{01_vector_add,02_shared_memory,03_bank_conflicts,04_tiled_gemm,05_tensor_core_mma}
mkdir -p benchmarks/{memory_bandwidth,compute_peak,roofline,occupancy_study}
mkdir -p projects/{mini-blas,image-filter-pipeline,parallel-reduction-lib}
mkdir -p scripts
mkdir -p data/profiling_logs
mkdir -p docs

# Root files
touch README.md LICENSE .gitignore

# Notes placeholders
touch notes/cuda/{memory-hierarchy.md,warp-execution.md,tensor-cores.md,occupancy-calculation.md,roofline-model.md}
touch notes/linux/{wsl-setup.md,perf-tools.md,kernel-basics.md}
touch notes/gpu-architecture/{sm-structure.md,rtx5070ti-analysis.md,bandwidth-calculations.md}
touch notes/research-log.md

# Experiment placeholders
for dir in experiments/*; do
    touch "$dir/README.md"
    touch "$dir/analysis.md"
done

# Benchmark placeholders
for dir in benchmarks/*; do
    touch "$dir/README.md"
    touch "$dir/analysis.md"
done

# Project placeholders
for dir in projects/*; do
    mkdir -p "$dir"/{src,include,tests}
    touch "$dir/CMakeLists.txt"
    touch "$dir/README.md"
done

# Scripts
touch scripts/{run_benchmark.sh,profile.sh,clean.sh}

# Docs
touch docs/{roadmap.md,learning-plan.md,references.md,performance-guidelines.md}

# Gitignore (CUDA + CMake + Linux focused)
cat <<EOF > .gitignore
# Build
build/
cmake-build-*/
CMakeFiles/
CMakeCache.txt
Makefile

# CUDA
*.cubin
*.ptx
*.fatbin

# Executables
*.out
*.exe
*.o

# Logs
*.log

# Python
__pycache__/
*.pyc

# Data outputs
data/*.csv
data/profiling_logs/*

# VSCode
.vscode/
EOF

# Basic README template
cat <<EOF > README.md
# mdameenh Systems Lab

## Focus Areas
- CUDA Programming
- GPU Architecture
- Memory Bandwidth Analysis
- Roofline Modeling
- Performance Engineering

## Environment
- Windows 11 + WSL2
- Ubuntu 22.04
- CUDA Toolkit
- RTX 5070 Ti

## Structure
- notes/        -> Technical notes and derivations
- experiments/  -> Isolated CUDA experiments
- benchmarks/   -> Performance measurement studies
- projects/     -> Larger integrated systems
- scripts/      -> Automation tools
- data/         -> Benchmark outputs
- docs/         -> Roadmap and planning
EOF

echo "Repository structure created successfully."
echo "Next steps:"
echo "1. cd $REPO_NAME"
echo "2. git init"
echo "3. git add ."
echo "4. git commit -m 'Initial structured systems lab setup'"