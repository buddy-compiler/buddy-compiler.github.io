---
layout: default
title: Open Projects
nav_order: 2
---

# Open Projects

Currently, we provide open projects in the following directions.

## 1 - Buddy Compiler Frontend

### Project 1-1: Add Operation Mappings in Dynamo Compiler

- **Description**: Expand the Dynamo Compiler's registry of operation mappings to ensure a more comprehensive encapsulation of the graph capture and Aten IR in TorchDynamo.

### Project 1-2: Transition to Dynamo Compiler for MLIR Deep Learning Models

- **Description**: The current buddy-benchmark repository hosts a collection of [MLIR Deep Learning model files](https://github.com/buddy-compiler/buddy-benchmark/tree/main/benchmarks/DeepLearning/Models). This project aims to supplant these existing explicit MLIR Deep Learning model files with those compiled using the Dynamo Compiler, thereby streamlining and updating the benchmark suite.

## 2 - Buddy Compiler Midend

Analyze and optimize workloads of various AI models and multimodal processes to improve operation efficiency on multiple backend platforms.

### Project 2-1: Operations Optimization for CPU SIMD Platforms

- **Description**: This project involves multi-level optimizations with the Vector Dialect, Affine Dialect, Transform Dialect, etc., targeting the platforms that support X86 AVX and Arm Neon instruction sets.

### Project 2-2: Operations Optimization for CPU Vector (RVV) Platforms

- **Description**: This project involves multi-level optimizations with the Vector Dialect, Affine Dialect, Transform Dialect, etc., targeting the platforms that support RVV 1.0 instruction set.

### Project 2-3: Operations Optimization for GPU Platforms

- **Description**: This project involves multi-level optimizations with the GPU Dialect, Vector Dialect, Affine Dialect, Transform Dialect, etc., targeting the GPU platforms.

### Project 2-4: Operations Optimization for Gemmini Accelerator

- **Description**: This project involves multi-level optimizations with the Gemmini Dialect, Affine Dialect, Transform Dialect, etc., targeting the Gemmini accelerator.

## 3 - Buddy Compiler Backend

### Project 3-1: Enhancement and Development of Gemmini LLVM Backend with JIT Execution Engine

- **Description**: This project aims to refine the existing Gemmini LLVM backend infrastructure and construct a robust Just-In-Time (JIT) execution engine. The engine is designed to effectively interpret and execute LLVM IR or dialects within the Gemmini simulator/evaluation platform, optimizing for high performance and efficiency.

## 4 - Buddy Compiler As A Service

[Buddy-CAAS Work Plan](https://buddycompiler.notion.site/Buddy-CAAS-Work-Plan-9e7eea61ddb04ea696599d904f2327a5)
