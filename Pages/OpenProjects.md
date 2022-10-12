---
layout: default
title: Open Projects
nav_order: 2
---

# Open Projects

Currently, we provide open projects in the following directions.

## 1 - Domain-specific Compiler

[Buddy Compiler DIP&DAP Work Plan](https://www.notion.so/buddycompiler/0e4297fd86774c3c9d53e461a933c4e7?v=3f4310b7121a4be88de719e7ac1a208c)

### Project 1-1: Adding a Compiler Stack for RISC-V Custom Extensions.

- **Description**: Both RISC-V and MLIR are modular and extensible, so we hope to use LLVM/MLIR to implement an integrated compiler stack infrastructure for RISC-V custom ISA.
- **Expected outcomes**: 
    - RISC-V custom ISA registration mechanism. 
    - Integrated MLIR/LLVM code generation mechanism.
    - The connection of the above two mechanisms.
- **Skills required**:  
    - Good C++ coding skills.
    - Basic understanding of MLIR and LLVM.
    - Basic understanding of RISC-V.
- **Possible mentors**: Hongbin Zhang
- **Difficulty rating**: Hard - Extremely Hard

### Project 1-2: Adding Morphological Transformations in DIP (Digital Image Processing) Dialect.

- **Description**: Seven morphological operations can be implemented for this project: Erosion, Dilation, Opening, Closing, Morphological Gradient, Top Hat, and Black Hat. 
- **Expected outcomes**: 1～2 morphological transformation operations and lowering passes.
- **Skills required**:  
    - Good C++ coding skills.
    - Basic understanding of MLIR.
    - Basic understanding of image processing.
- **Possible mentors**: Prathamesh Tagore
- **Difficulty rating**: Medium - Hard

### Project 1-3: Adding the Basic Support for PC (Point Cloud) Dialect.

- **Description**: This project intends to add the abstraction for point cloud at the IR level. Participants should add point cloud data structure, implement the PC dialect, and complete an end-to-end example.
- **Expected outcomes**: 
    - Data container for the point cloud.
    - Implement the PC dialect.
    - Choose an operation and implement the lowering pass.
    - Complete an end-to-end example.
    - Add a benchmark for the corresponding operation.
- **Skills required**:  
    - Good C++ coding skills.
    - Basic understanding of MLIR.
    - Basic understanding of point cloud.
- **Possible mentors**: Hongbin Zhang
- **Difficulty rating**: Medium - Hard

## 2 - Performance Optimization

### Project 2-1: Vectorizing Convolution or GEMM Operations.

- **Description**: Optimize existing convolution and GEMM operations using vectorization. Participants can refer to the algorithm [here](https://github.com/opencv/opencv/blob/4.x/modules/dnn/src/layers/layers_common.simd.hpp).
- **Expected outcomes**: 
    - Add vectorization passes for convolution or GEMM operations.
    - Add a benchmark for the optimization pass.
- **Skills required**: 
    - Good C++ coding skills.
    - Basic understanding of MLIR.
    - Basic understanding of vectorization.
- **Possible mentors**: Liutong Han, Hongbin Zhang
- **Difficulty rating**: Hard - Extremely Hard

## 3 - Benchmark Framework

### Project 3-1: Adding More Deep Learning Benchmark Cases and Items in buddy-benchmark.

- **Description**: There are already some deep learning benchmarks in buddy-benchmark. This project hopes to add more comparison deep learning compiler toolchains (tvm, iree, onnx-mlir, torch-mlir, etc.) and comparison items (e.g., peak memory allocation) on this basis.
- **Expected outcomes**: 
    - Add more cases to the model level benchmark.
    - Add model level benchmark for tvm, iree, onnx-mlir, torch-mlir, etc.
    - Add more comparison items (e.g., peak memory allocation)
- **Skills required**: 
    - Some deep learning compiler experience.
    - Basic understanding of MLIR.
- **Possible mentors**: Hongbin Zhang
- **Difficulty rating**: Medium - Hard

## 4 - Testing Framework

### Project 4-1: Improving the Testing Framework for buddy-mlir and buddy-benchmark.

- **Description**: The buddy compiler (buddy-mlir and buddy-benchmark) contains many levels of work (IR, API, runtime, etc.), and we need to test each level to ensure correctness. Currently, there are some basic tests in the project, and this project needs to add more test cases and design automated test methods.
- **Expected outcomes**:
    - Add more test cases for each level of work.
    - Design automated test methods.
- **Skills required**:
    - Good C++ coding skills.
    - Basic understanding of MLIR.
    - Basic software testing knowledge and experience.
- **Possible mentors**: Hongbin Zhang, Prathamesh Tagore
- **Difficulty rating**: Easy - Medium

## 5 - Buddy Compiler As A Service

[Buddy-CAAS Work Plan](https://buddycompiler.notion.site/Buddy-CAAS-Work-Plan-9e7eea61ddb04ea696599d904f2327a5)
