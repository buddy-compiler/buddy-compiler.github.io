---
layout: default
title: Open Projects
nav_order: 2
---

# Open Projects

We are excited to invite passionate individuals to join us.
If you are interested in making meaningful open-source contributions and gaining invaluable experience,
contact us to become part of our team in this journey.

## 1 - Buddy Compiler Frontend

### Project 1-1: Add Operation Mappings in Graph Representation

- **Description:** Expand the Buddy Compiler’s graph representation registry of operation mappings to ensure a more comprehensive encapsulation of the PyTorch and ONNX interfacing. The focus is to improve the compiler's ability to optimize and execute a broader range of models efficiently.

- **Expected Outcomes:**
    - Enhanced integration of PyTorch AtenIR operations, boosting Dynamo Compiler's functionality and efficiency.
    - Integration of ONNX operations to support a broader range of model architectures.
    - Comprehensive support for models including but not limited to LLaMA, Bert, CLIP, Whisper, Stable Diffusion, ResNet, and MobileNet.
    - Replacement of [static deep learning models](https://github.com/buddy-compiler/buddy-benchmark/tree/main/benchmarks/DeepLearning/Models) in the buddy-benchmark repository with ones utilizing the Buddy Compiler frontend.

- **Skills Required:**
    - Solid experience with PyTorch and ONNX frameworks.
    - Proficiency in MLIR and Python bindings.
    - A strong understanding of compiler design principles and IR.

- **Possible Mentors:** Linquan Wei, Yuliang Li
- **Expected Size of Project:** 175 hour
- **Rate:** Easy

## 2 - Buddy Compiler Midend

Analyze and optimize workloads of various AI models and multimodal processes to improve operation efficiency on multiple backend platforms.

### Project 2-1: Operations Optimization for CPU SIMD Platforms

- **Description:** This project involves multi-level optimizations with the Vector Dialect, Affine Dialect, Transform Dialect, etc., targeting the platforms that support X86 AVX and Arm Neon instruction sets.

- **Expected Outcomes:**
    - Implement optimization passes in the buddy-mlir repository for deep learning workloads and operations targeting SIMD platforms.
    - Conduct performance comparisons at both the operation level and the model level within the buddy-benchmark repository.
    - Achieve comparable performance with other optimization frameworks such as TVM and IREE for targeted platforms.

- **Skills Required:**
    - Proficiency in MLIR infrastructure and C++ programming.
    - In-depth understanding of optimizations for CPU SIMD platforms.
    - Familiarity with the design methodologies and optimization strategies of TVM and IREE.

- **Possible Mentors:** Xulin Zhou, Liutong Han, Hongbin Zhang
- **Expected Size of Project:** 350 hour
- **Rate:** Hard

### Project 2-2: Operations Optimization for GPU Platforms

- **Description**: This project involves multi-level optimizations with the GPU Dialect, Transform Dialect, Vector Dialect, Affine Dialect, etc., targeting the GPU platforms.
- **Expected Outcomes:**
    - Implement optimization passes in the buddy-mlir repository for deep learning workloads and operations targeting GPU platforms.
    - Conduct performance comparisons on GPU platforms at both the operation level and the model level within the buddy-benchmark repository.
    - Achieve comparable performance with other optimization frameworks such as Triton, TVM, and IREE for targeted platforms.
- **Skills Required:**
    - Proficiency in MLIR infrastructure and C++ programming.
    - In-depth understanding of optimizations for GPU platforms.
    - Familiarity with the design methodologies and optimization strategies of Triton, TVM, and IREE.
- **Possible Mentors:** Zikang Liu
- **Expected Size of Project:** 350 hour
- **Rate:** Hard

## 3 - Buddy Compiler Backend

### Project 3-1: Enhancement and Development of Gemmini LLVM Backend with JIT Execution Engine

- **Description**: [Gemmini](https://github.com/ucb-bar/gemmini) is a systolic array accelerator. This project aims to refine the existing Gemmini LLVM backend infrastructure and construct a robust Just-In-Time (JIT) execution engine in buddy-mlir repository. The engine is designed to effectively interpret and execute LLVM IR or dialects within the Gemmini simulator/evaluation platform, optimizing for high performance and efficiency.
- **Expected Outcomes:**
    - Enhance the optimization capabilities of the Gemmini backend and refine the Gemmini compilation pipeline.
    - Develop a JIT execution engine and runtime libraries specifically for Gemmini.
    - Conduct performance comparisons at both the operation and model levels within the buddy-benchmark repository, ensuring competitive performance with Gemmini software stack.
- **Skills Required:**
    - Proficiency in MLIR and LLVM infrastructure and C++ programming.
    - In-depth understanding of existing Gemmini LLVM backend in buddy-mlir repository.
    - Familiarity with the Gemmini accelerator.
- **Possible Mentors:** Zhongyu Qin, Hongbin Zhang
- **Expected Size of Project:** 175 hour
- **Rate:** Intermediate
