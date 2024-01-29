---
layout: default
title: Chapter 0 - MLIR Primer
parent: Tutorial
---

## MLIR Primer

### IR Structure 

<!-- 

Reference:
https://mlir.llvm.org/docs/LangRef/#high-level-structure
https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/

1. 数据结构
MLIR 基于一种“类图”的数据结构构造，节点叫做 Operation，边叫做 Value.
每个 Value 是一个 Operation 或者 Block Argument 的结果.
Operation / Block / Region 的关系.
数据结构是循环嵌套的关系.

2. 语义表达
Operatoin 是 MLIR 语义表达的载体.
MLIR 使用 Traits 和 Interface 表示 Operation 的抽象语义.
Traits 主要负责...
Interface 主要负责...
MLIR 可以表示 SSA-based IR（如，LLVM IR），也可以表示抽象语法树，以及硬件特定指令，或者硬件电路.

3. IR 遍历与变换

使用 `-test-print-nesting` 遍历嵌套 IR.
使用 getOps<OpTy>() 可以获取 Block 或者 Rigion 里面的每一个目标 Operation，可用于遍历单个 Block 或者 Region.
Operation / Block / Region 的 walk() helper function，针对特定 Operation 执行回调函数
遍历 def-use chain，通过各种 `getXXX` 函数获得 Value 或者 Operation 从而通过调用关系遍历 IR（例如，`-test-print-defuse` 可作为 demo）

------

MLIR 提供可扩展的框架对 Operation 进行变换，也就是构造 Pass
Pass 通常以一个 operation 为根结点.

MLIR（多级中间表示）是一种编译器中间表示，它结合了传统三地址SSA表示（如LLVM IR或SIL）的特点，并引入了多面体循环优化中的概念作为一级概念。
MLIR的这种混合设计旨在优化表示、分析和转换高级数据流图以及为高性能数据并行系统生成的特定目标代码。
MLIR不仅在表示能力上独特，其单一连续的设计还提供了一个框架，用于从数据流图降低到高性能特定目标代码。

MLIR的高级结构基于图形数据结构，由称为“操作（Operations）”的节点和称为“值（Values）”的边组成。
每个值都是一个操作或块参数的结果，并且具有由类型系统定义的值类型。
操作包含在块中，块包含在区域中。
操作在其包含块内有序，块在其包含区域内有序，尽管这个顺序在给定类型的区域中可能有或没有语义意义。
操作还可以包含区域，使得可以表示层次结构。

操作可以代表许多不同的概念，从高级概念（如函数定义、函数调用、缓冲区分配、缓冲区视图或切片以及进程创建）到低级概念（如目标无关的算术、特定目标的指令、配置寄存器和逻辑门）。
这些不同的概念由MLIR中的不同操作表示，MLIR中可用的操作集可以任意扩展。

MLIR还提供了一个可扩展的框架，用于对操作进行转换，使用熟悉的编译器“传递（Passes）”概念。
在任意一组操作上启用任意一组传递会带来显著的扩展挑战，因为每个转换都必须考虑任何操作的语义。
MLIR通过允许使用特征（Traits）和接口（Interfaces）抽象地描述操作语义来解决这种复杂性，使转换能够更通用地操作操作。
特征通常描述了有效IR的验证约束，使复杂的不变量能够被捕获和检查。

MLIR的一个明显应用是表示基于SSA的IR，如LLVM核心IR，通过选择适当的操作类型来定义模块、函数、分支、内存分配，并验证约束以确保SSA支配属性。
MLIR包括一系列定义了这种结构的方言。
然而，MLIR旨在足够通用，以表示其他编译器类似的数据结构，如语言前端中的抽象语法树、特定目标后端中生成的指令，或高级综合工具中的电路。

MLIR的IR结构通过示例进行说明，并同时介绍了操作它的C++ API。
MLIR的IR是递归嵌套的，一个操作可以有一个或多个嵌套区域，每个区域实际上是一个块的列表，每个块本身包装了一个操作列表。
这种结构的遍历将遵循三种方法：printOperation()、printRegion()和printBlock()。

总的来说，MLIR提供了一个灵活且强大的框架，用于表示和处理各种级别的编译器中间表示，从高级数据流图到特定目标的代码生成。
通过其独特的设计，MLIR能够有效地处理复杂的编译器优化和转换任务。
 -->
