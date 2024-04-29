---
layout: default
title: Home
nav_order: 1
description: "Buddy compiler is a domain-specific compiler infrastructure."
permalink: /
---

# Buddy Compiler

Buddy Compiler is a domain-specific compiler infrastructure. We use "buddy" as the name because we want to build a buddy system to help users easily design, implement, and evaluate domain-specific compilers. 

Buddy Compiler community is an open source community, where we explore intriguing features and implement ideas for compiler infrastructure by working together. Join us through [this slack link](https://join.slack.com/t/buddycompiler/shared_invite/zt-13y6ibj4j-n6MQ8u9yCUPltCCDhLEmXg) if you want to contribute.

Buddy Compiler As A Service (Buddy-CAAS) is an online platform that serves users and developers to configure the pass pipeline and demonstrate on multiple backends quickly and smoothly. Click [here](https://buddy.isrc.ac.cn/) to explore and experiment with our online platform. For more details, please see [Buddy-CAAS documentation](./Pages/Documentation/CAAS.md).

[GitHub](https://github.com/buddy-compiler){: .btn .btn-primary .fs-4 .mb-4 .mb-md-0 .mr-1 }
<!-- [Projects](https://buddycompiler.notion.site/7f92ee739453461d956b5b4e4bb73bf1?v=70f2180e94ce4f7fa5bac01f4b47b98e&pvs=4){: .btn .btn-primary .fs-4 .mb-4 .mb-md-0 .mr-1 } -->
<!-- [Tasks](https://buddycompiler.notion.site/3f4b8e480d6e447dbd4c3b3e21fa1208?v=8dc8526cba2245a98a726be1c08e0f6b&pvs=4){: .btn .btn-primary .fs-4 .mb-4 .mb-md-0 .mr-1 } -->
<!-- [Team](https://buddycompiler.notion.site/c912d8237b79409c89cf46b432b6a3ca?v=186d4ed4b9fa4452a08ca5af0921208a&pvs=4){: .btn .btn-primary .fs-4 .mb-4 .mb-md-0 .mr-1 } -->

## Motivation

Our goal is to address the challenges of combining domain-specific languages (DSLs) and domain-specific architectures (DSAs) by providing our compiler-level infrastructure, Buddy Compiler. Built on top of [MLIR](https://mlir.llvm.org/) and [RISC-V](https://riscv.org/), Buddy Compiler aims to create a unified ecosystem that unlocks more software-hardware co-design opportunities. Such ecosystem can simplify the development process and accelerate optimize performance, making it easy for users to develop their own compilers.

With the rapid development of applications demanded for high computing power, general-purpose processors can not meet computing needs in the context of expiration of Moore’s Law. By utilizing DSLs and DSAs, developers can leverage the benefits of both to optimize the performance and efficiency of the computations within a targeted domain. However, Combining DSLs and DSAs presents some challenges in various aspects, including development complexity, performance trade-offs, and tooling support. Our goal is to provide an infrastructure at the compiler level to explore solutions to these problems.

The combined DSL-DSA solutions involve various software-hardware co-design techniques. We claim that a unified ecosystem can get more opportunities for co-design, and we particularly embrace the MLIR and RISC-V ecosystem. MLIR is a revolutionary multi-level intermediate representation and compiler infrastructure that provides reusable and extensible mechanisms. RISC-V is an open-sourced instruction set architecture with a modular design for custom extensions. Both MLIR and RISC-V have extensible concepts to maximize the reuse of base parts, which is suitable for domain-specific design. Buddy Compiler is based on MLIR and has specific support for RISC-V, especially with respect to vectorization.

For more information, please see our [documents](https://github.com/buddy-compiler/buddy-mlir/tree/main/docs) and [open projects](./Pages/OpenProjects.md).

## Overview

Currently, the buddy compiler contains the following two modules:

- buddy-mlir (get started from [here](https://github.com/buddy-compiler/buddy-mlir))

The buddy-mlir is the main framework of Buddy Compiler. We use MLIR as the cornerstone and explore how to build a domain-specific compiler on top of it. Our research in this framework includes DSL frontend support, IR-level optimization, DSA backend code generation, MLIR-related development tools, etc.

- buddy-benchmark (get started from [here](https://github.com/buddy-compiler/buddy-benchmark))

The buddy-benchmark is a benchmark framework to evaluate domain-specific compilers and libraries. Evaluation is an essential step in developing a compiler. We can hardly find a unified benchmark to evaluate compiler or optimization in some domains. We thus propose an extensible benchmark framework to collect domain-specific evaluation cases.

The graph below shows the modules of the buddy compiler.

![overview](./Images/overview.png)

## Publications and Presentations

- Compiler Technologies in Deep Learning Co-Design: A Survey - [Link](https://spj.science.org/doi/10.34133/icomputing.0040)
- AutoConfig: A Configuration-Driven Automated Mechanism for Deep Learning Compilation Optimization - [Link](https://www.jos.org.cn/jos/article/abstract/7102)
- Buddy Compiler @ CGO C4ML Workshop 2024 - [Poster](https://github.com/buddy-compiler/buddy-compiler.github.io/blob/master/Resources/BuddyCompiler%40C4ML2024.pdf) / [Link](https://www.c4ml.org/)
- Buddy Compiler @ EuroLLVM 2023
    - Buddy Compiler: An MLIR-based Compilation Framework for Deep Learning Co-design - [Link](https://www.youtube.com/watch?v=EELBpBA-XCE)
    - RISC-V Vector Extension Support in MLIR: Motivation, Abstraction, and Application - [Link](https://www.youtube.com/watch?v=i9dsjzVOvy8)
    - Image Processing Ops as first class citizens in MLIR: write once, vectorize everywhere! - [Link](https://www.youtube.com/watch?v=0xQ2lDY9RCw)
    - Buddy-CAAS: Compiler As A Service for MLIR - [Link](https://www.youtube.com/watch?v=f7USv-oAtvI)


## Next Steps

If you are interested in our project, you can play around with examples in [buddy-mlir](https://github.com/buddy-compiler/buddy-mlir) and [buddy-benchmark](https://github.com/buddy-compiler/buddy-benchmark). Then you can see if there are [projects in the list](./Pages/OpenProjects.md) that appeal to you; feel free to contact us via [slack](https://join.slack.com/t/buddycompiler/shared_invite/zt-13y6ibj4j-n6MQ8u9yCUPltCCDhLEmXg) for more details. We also provide a [contributor guide](./Pages/ContributorGuide.md) for you if you want to contribute your code.
