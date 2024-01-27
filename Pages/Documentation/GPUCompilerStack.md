---
layout: default
title: Collaboration System Guide
parent: Documentation
---

# GPUCompilerStack

This article will introduce in detail the GPU compilation stack of the mainstream deep learning compilers.

## MLIR

We will give a detailed introduction to the passes needed to lower to gpu mlir in MLIR in this part.

### `-gpu-kernel-outlining`

The pass -gpu-kernel-outlining is an MLIR pass for optimizing GPU kernel functions, which can unfold the loops in GPU kernel functions into multiple loops, thereby improving the execution efficiency and readability of GPU kernel functions.

This pass can effectively reduce unnecessary or redundant operations in GPU kernel functions, and improve the call frequency between different sub-functions in GPU kernel functions. This can improve the performance of GPU kernel functions when running on different types and sizes of GPUs.

The pass can be created by `createGpuKernelOutliningPass()`(This code can by located in `lib/Dialect/GPU/Transforms/KernelOutlining.cpp`). 

```c++
std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGpuKernelOutliningPass(StringRef dataLayoutStr) {
  return std::make_unique<GpuKernelOutliningPass>(dataLayoutStr);
}

class GpuKernelOutliningPass
    : public impl::GpuKernelOutliningBase<GpuKernelOutliningPass> {
public:
  GpuKernelOutliningPass(StringRef dlStr) {
    if (!dlStr.empty() && !dataLayoutStr.hasValue())
      dataLayoutStr = dlStr.str();
  }

  GpuKernelOutliningPass(const GpuKernelOutliningPass &other)
      : GpuKernelOutliningBase(other), dataLayoutSpec(other.dataLayoutSpec) {
    dataLayoutStr = other.dataLayoutStr.getValue();
  }

  LogicalResult initialize(MLIRContext *context) override {
    if (!dataLayoutStr.empty()) {
      Attribute resultAttr = mlir::parseAttribute(dataLayoutStr, context);
      if (!resultAttr)
        return failure();

      dataLayoutSpec = dyn_cast<DataLayoutSpecInterface>(resultAttr);
      if (!dataLayoutSpec)
        return failure();
    }

    return success();
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<SymbolOpInterface>()) {
      Block::iterator insertPt(func->getNextNode());
      auto funcWalkResult = func.walk([&](gpu::LaunchOp op) {
        SetVector<Value> operands;
        std::string kernelFnName =
            Twine(op->getParentOfType<SymbolOpInterface>().getName(), "_kernel")
                .str();

        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(op, kernelFnName, operands);

        auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
        symbolTable.insert(kernelModule, insertPt);

        convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
        modified = true;
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();
    }

    if (modified)
      getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                              UnitAttr::get(&getContext()));
  }

private:
  gpu::GPUModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable) {
    auto *context = getOperation().getContext();
    OpBuilder builder(context);
    auto kernelModule = builder.create<gpu::GPUModuleOp>(kernelFunc.getLoc(),
                                                         kernelFunc.getName());

    if (dataLayoutSpec)
      kernelModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayoutSpec);

    SymbolTable symbolTable(kernelModule);
    symbolTable.insert(kernelFunc);

    SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (std::optional<SymbolTable::UseRange> symbolUses =
              SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName =
              cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
          if (symbolTable.lookup(symbolName))
            continue;

          Operation *symbolDefClone =
              parentSymbolTable.lookup(symbolName)->clone();
          symbolDefWorklist.push_back(symbolDefClone);
          symbolTable.insert(symbolDefClone);
        }
      }
    }

    return kernelModule;
  }

  Option<std::string> dataLayoutStr{
      *this, "data-layout-str",
      llvm::cl::desc("String containing the data layout specification to be "
                     "attached to the GPU kernel module")};

  DataLayoutSpecInterface dataLayoutSpec;
};

```

The implementation details of the above code are as follows: 


1. Invoke `getOperation()` to obtain all the Operations in the current context,
2. Then traverse these ops, and create the corresponding `GPUFuncOp` according to the current op using the lambda expression `funcWalkResult`.
3. `funcWalkResult` will generate the kernelName that corresponds to the current op. The `outlineKernelFuncImpl` will create the relevant `GPUFuncOp` (like `gpu.func @matmul_kernel`) based on the kernelName and op.
4. `outlineKernelFuncImpl` will create the suitable `OpBuilder` according to the op's context, `OpBuilder` will create the relevant `gpu.func` according to the parameter type, and set the proper block and grid size. Then it will map the original op's attributes, parameters and region to the suitable position of the created gpu.func.
5. Invoke createKernelModule, this method will create the relevant gpu.module according to the `GPUFuncOp` (like `gpu.module @matmul_kernel`)
6. Invoke `convertToLaunchFuncOp` to produce the relevant `gpu.launch_func`.

### `-gpu-map-parallel-loops`

`-gpu-map-parallel-loops` driver for implementing parallel loop optimization on the GPU. It can convert a loop into multiple parallel loops and select an appropriate parallelization strategy (such as number of threads, inter-thread communication or data allocation) based on different goals (such as performance, power consumption or memory usage). 

The implementation of this pass is in `lib/Dialect/GPU/Transforms/ParallelLoopMapper.cpp`. When using it, call the `createGpuMapParallelLoopsPass()` function to create the pass. This function will return a `GpuMapParallelLoopsPass` structure.

```c++
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::createGpuMapParallelLoopsPass() {
  return std::make_unique<gpu::GpuMapParallelLoopsPass>();
}

struct GpuMapParallelLoopsPass
    : public impl::GpuMapParallelLoopsPassBase<GpuMapParallelLoopsPass> {
  void runOnOperation() override {
    for (Region &region : getOperation()->getRegions()) {
      region.walk([](ParallelOp parallelOp) { mapParallelOp(parallelOp); });
    }
  }
};
```

This method will traverse all regions and execute mapParallelOp on the walker of each region.Enumeration class `MappingLevel`, representing three mapping levels: `Grid`, `Block`, `Sequential`.

```c++
enum MappingLevel { MapGrid = 0, MapBlock = 1, Sequential = 2 };
```

Each mapping level is divided into three levels: x, y, and z according to the input dimensions.
`mapParallelOp` maps parallel loops to corresponding workgroups. The default MappingLevel of `mapParallelOp` is `Grid`. The first cycle encountered is mapped to the `Grid level`, the second cycle encountered is mapped to the `Block level`, and so on.

### `-convert-parallel-loops-to-gpu`

This pass is to convert `scf.parallel` to `gpu.launch` operation.This pass is implemented by `createParallelLoopToGpuPass()` (the code is located in `lib/Conversion/SCFToGPU/SCFToGPUPass.cpp`).


```c++
std::unique_ptr<Pass> mlir::createParallelLoopToGpuPass() {
  return std::make_unique<ParallelLoopToGpuPass>();
}
```


In `ParallelLoopToGpuPass`, the `scf.parallel` to `gpu.launch` operation is mainly implemented through the `matchAndReWrite` method of `ParallelToGpuLaunchLowering`. The main idea of this method:

1. Check whether the operation has a parent operation of type `ParallelOp`. If so, it means that the operation is not the outermost loop, then a failure value is returned because it is not supported to initiate a GPU kernel inside a parallel loop.

2. Created a `gpu::LaunchOp` operation, which is a class that represents an operation that launches a GPU core on the specified grid and thread block. It uses a constant 1 as the size of all grids and thread blocks, and these sizes will be adjusted later according to the mapping relationship. A single region containing the kernel body is also created, and a `gpu::TerminatorOp` operation is inserted at the end of the region, which is a class that represents the termination operation of a GPU kernel. It then sets the insertion point to the beginning of the range.

3. An `IRMapping` object is created, which represents a mapping from operations and values to operations and values, which is used to maintain consistency during cloning operations. An `llvm::DenseMap` object is created, representing a mapping from `gpu::Processor` to Value to store the grid and thread block sizes.

4. Call `processParallelLoop` to process a `ParallelOp` operation and convert it into a `gpu::LaunchOp` operation, while updating the mapping relationship, work list, grid and thread block size.

5. Use a loop to iterate through the work list until the work list is empty. In each loop, it does the following:

   a. It pops an operation from the work list and handles it differently depending on the type of operation.
   
   b. If the operation is a `ParallelOp` type operation, which means the operation is a nested parallel loop, then do the following:

     ⅰ. Before entering a nested scope, check whether there have been any side effects. If so, return a failure value because nested parallel loops in code with side effects are not supported.

     ⅱ. Call the `processParallelLoop` function to process this nested parallel loop and update the mapping relationship, work list, grid and thread block size. If this function returns a failure value, then a failure value is returned.

     ⅲ. If the operation is a `gpu::LaunchOp` type operation, it means that the operation is a previously created launch operation. It serves as a sentinel value to indicate that a nested level operation has been completed, then do the following:

     ⅳ. Get the parent operation of the current insertion point, and set the insertion point behind the parent operation, so that you can continue to process the operation of the previous level.

     ⅴ. Set the variable leaving the nested scope to true, indicating that it is no longer in the innermost scope.

     vi. Set the variable that sees the side effect to false, indicating that the status of the side effect is reset.

   c. Otherwise, it means that the operation is a common operation, then do the following:

     ⅰ. Use the rewriter and mapping relationship to clone this operation, and map the result of the operation to the cloned result so that consistency can be maintained.

     ⅱ. Check whether the cloning operation has side effects or whether there are areas. If so, set the variable that sees the side effect to true, indicating that we need to pay attention to the impact of the side effect.

     iii. If the innermost scope has been left and a side effect is seen, a failure value is returned, since side effects in non-innermost scopes are not supported.

6. After the loop ends, it means that the startup operation has been successfully created, then do the following:

   a. Traverse the mapping of grid and thread block sizes, obtain the number of the operand of the startup operation according to the mapped key (value of gpu::Processor type), and then set the operand of the startup operation to the mapped value (Value type value) so that the grid and thread block size of the initiated operation can be updated.

   b. Use the rewriter to remove the original parallel loop operation since it has been converted to a startup operation.

### `-convert-gpu-to-nvvm`

Use `-convert-gpu-to-nvvm` pass to convert gpu dialect to nvvm dialect. Through the conversion of this pass, the final llvm ir can run on the NVIDIA GPU. In mlir, this pass is implemented through `struct LowerGpuOpsToNVVMOpsPass` (the code is located at: `lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp`).

This pass will replace all GPU device operations that occur with corresponding nvvm operations. This pass will only process device code!

```c++
struct LowerGpuOpsToNVVMOpsPass
    : public impl::ConvertGpuOpsToNVVMOpsBase<LowerGpuOpsToNVVMOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(&getContext()));
    }
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.useBarePtrCallConv = useBarePtrCallConv;
    {
      RewritePatternSet patterns(m.getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }

    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kGlobalMemorySpace);
          case gpu::AddressSpace::Workgroup:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kSharedMemorySpace);
          case gpu::AddressSpace::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    RewritePatternSet llvmPatterns(m.getContext());

    arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    if (this->hasRedux)
      populateGpuSubgroupReduceOpLoweringPattern(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

```

The main implementation methods of this pass:

1. Get the current operating gpu module through `gpu::GPUModuleOp m = getOperation()`;

2. Create a `LowerToLLVMOptions` object options, which is used to store some lowering operations, such as whether to use native pointer calls and whether to use index calculation.

3. Create an `LLVMTypeConverter` object and initialize it according to options. This object is used to convert the GPU type to the LLVM type.

4. Call `populateGpuMemorySpaceAttributeConversions` to convert the memory space attributes, and convert the memory space attributes of the gpu into the corresponding nvvm memory space attributes.

5. Call the `populateGpuToNVVMConversionPatterns` method to convert the GPU to `NVVMOps` type. This method will modify or add the GPU to NVVMOps type stored in the converter according to the given llvmPatterns.

6. Call `populateGpuWMMAToNVVMConversionPatterns` to convert `GPUWMMAType` to `NVVMOps` type.

7. When using `-convert-gpu-to-nvvm again`, we will add `-convert-gpu-to-nvvm(has_redux='1')`. If there is Redux, call `populateGpuSubgroupReduceOpLoweringPattern` to downgrade the sub-combination operation under the Redux function.

8. Create `LLVMConversionTarget` and initialize it using getContext.

9. Use `configureGpuToNVVMConversionLegality(target)` to check whether the GPU to `NVVMOps` type conversion is legal.

10. Call `applyPartialConversion` to perform partial conversion. If the conversion fails, an exception will be thrown.

### `-convert-nvvm-to-llvm`

Use this pass to convert nvvm ir to llvm ir. This passmlir provides the ability to process CUDA code under the unified framework.

This pass is mainly implemented through `ConvertNVVMToLLVMPass` (the code is located at: `lib/Conversion/NVVMToLLVM/NVVMToLLVM.cpp`).

```c++
struct ConvertNVVMToLLVMPass
    : public impl::ConvertNVVMToLLVMPassBase<ConvertNVVMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    RewritePatternSet pattern(&getContext());
    mlir::populateNVVMToLLVMConversionPatterns(pattern);
    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};
```

`mlir::populateNVVMToLLVMConversionPatterns(pattern)` adds `Ptxlowering` to `RewritePatternSet` and builds PTX assembly through `PtxLowering`.

```c++
struct PtxLowering
    : public OpInterfaceRewritePattern<BasicPtxBuilderInterface> {
  using OpInterfaceRewritePattern<
      BasicPtxBuilderInterface>::OpInterfaceRewritePattern;

  PtxLowering(MLIRContext *context, PatternBenefit benefit = 2)
      : OpInterfaceRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(BasicPtxBuilderInterface op,
                                PatternRewriter &rewriter) const override {
    if (op.hasIntrinsic()) {
      LLVM_DEBUG(DBGS() << "Ptx Builder does not lower \n\t" << op << "\n");
      return failure();
    }

    SmallVector<std::pair<Value, PTXRegisterMod>> asmValues;
    LLVM_DEBUG(DBGS() << op.getPtx() << "\n");
    PtxBuilder generator(op, rewriter);

    op.getAsmValues(rewriter, asmValues);
    for (auto &[asmValue, modifier] : asmValues) {
      LLVM_DEBUG(DBGSNL() << asmValue << "\t Modifier : " << &modifier);
      generator.insertValue(asmValue, modifier);
    }

    generator.buildAndReplaceOp();
    return success();
  }
};
```

The main idea of the above matchAndRewrite implementation is:

1. Create a SmallVector to store `pair<Value, PTXRegisterMod>`. Value represents an SSA value and PTXRegisterMod represents a register read/write modifier.

2. Create a `PTXBuilder` object, which can be used to build PTX assembly.

3. The `getAsmValues` method is called, which fills the `SmallVector` with values and modifiers that are used as operands for PTX assembly. It then iterates through the `SmallVector` and inserts each value and modifier into a PtxBuilder object.

4. Call `buildAndReplaceOp` of the `PtxBuilder` object to build an inline assembly operation and use it to replace or delete the original operation.

### `-gpu-to-llvm`

`-gpu-to-llvm` pass is a backend for converting GPU modules to LLVM IR. This backend can convert GPU-specific operations and data types in the GPU module to corresponding operations and data types in LLVM IR, thereby enabling the compilation and optimization of GPU code on LLVM.

This backend has the following main steps:
● It will traverse all functions and variables in the GPU module and map them to the corresponding functions and variables in LLVM IR. For example, it will map the gpu.alloc function to the llvm.memcpy function, the gpu.launch_func function to the llvm.launch function, etc.

● Then, it converts operations that use special formats or conventions in the GPU module. For example, it will convert `gpu.all_reduce min uniform { } : (f64) -> f64` to `llvm.sdiv_f32 min %0, %1 : f32 -> f32 etc`.

● Finally, it will convert the data types using special formats or conventions in the GPU module. For example, it will convert `gpu.spmat_get_size (gpu::SpMatGetSizeOp)` to `llvm.i32` getelementptr inbounds `i32, i32, i32, i32 0, 0, 0, 0 : i32 -> i32`.

This Pass is implemented through `GpuToLLVMConversionPass` (the code is located at: `lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp`).

```c++
void GpuToLLVMConversionPass::runOnOperation() {
  LowerToLLVMOptions options(&getContext());
  options.useBarePtrCallConv = hostBarePtrCallConv;

  LLVMTypeConverter converter(&getContext(), options);
  RewritePatternSet patterns(&getContext());
  LLVMConversionTarget target(getContext());

  SymbolTable symbolTable = SymbolTable(getOperation());
  // Preserve GPU modules if they have target attributes.
  target.addDynamicallyLegalOp<gpu::GPUModuleOp>(
      [](gpu::GPUModuleOp module) -> bool {
        return module.getTargetsAttr() != nullptr;
      });
  target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [&](gpu::LaunchFuncOp op) -> bool {
        auto module =
            symbolTable.lookup<gpu::GPUModuleOp>(op.getKernelModuleName());
        return converter.isLegal(op->getOperandTypes()) &&
               converter.isLegal(op->getResultTypes()) &&
               (module && module.getTargetsAttr() &&
                !module.getTargetsAttr().empty());
      });

  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                    target);
  populateGpuToLLVMConversionPatterns(converter, patterns, gpuBinaryAnnotation,
                                      kernelBarePtrCallConv, &symbolTable);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

The general idea of ​​implementing this code:
1. First is the initialization of variables. Initialize the following variables for subsequent conversion operations:

   a. Create an instance of LowerToLLVMOptions options(&getContext()), which specifies some conversion options

   b. Create LLVMTypeConverter converter(&getContext(), options) for type conversion between MLIR and LLVM IR

   c. Create a RewritePatternSet patterns(&getContext()); collection of rewrite patterns, which will store the patterns for converting gpu dialect to llvm dialect operations.

   d. Create LLVMConversionTarget target(getContext()) to specify the target platform for conversion
  
2. Use target.addDynamicallyLegalOp to determine whether the subsequent conversion operation is legal. The judgment method used is that only when the GPU module has the target attribute, it will be considered legal.

3. Call populateGpuToLLVMConversionPatterns to populate RewritePatternSet with conversion patterns for different types of GPU operations. The added conversion patterns are as shown in the following code:


```c++
void mlir::populateGpuToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns,
                                               StringRef gpuBinaryAnnotation,
                                               bool kernelBarePtrCallConv,
                                               SymbolTable *cachedModuleTable) {
  addOpaquePointerConversion<gpu::AsyncTokenType>(converter);
  addOpaquePointerConversion<gpu::SparseDnTensorHandleType>(converter);
  addOpaquePointerConversion<gpu::SparseSpMatHandleType>(converter);
  addOpaquePointerConversion<gpu::SparseSpGEMMOpHandleType>(converter);

  patterns.add<ConvertAllocOpToGpuRuntimeCallPattern,
               ConvertDeallocOpToGpuRuntimeCallPattern,
               ConvertHostRegisterOpToGpuRuntimeCallPattern,
               ConvertHostUnregisterOpToGpuRuntimeCallPattern,
               ConvertMemcpyOpToGpuRuntimeCallPattern,
               ConvertMemsetOpToGpuRuntimeCallPattern,
               ConvertSetDefaultDeviceOpToGpuRuntimeCallPattern,
               ConvertWaitAsyncOpToGpuRuntimeCallPattern,
               ConvertWaitOpToGpuRuntimeCallPattern,
               ConvertAsyncYieldToGpuRuntimeCallPattern,
               ConvertCreateDnTensorOpToGpuRuntimeCallPattern,
               ConvertDestroyDnTensorOpToGpuRuntimeCallPattern,
               ConvertCreateCooOpToGpuRuntimeCallPattern,
               ConvertCreateCooAoSOpToGpuRuntimeCallPattern,
               ConvertCreateCsrOpToGpuRuntimeCallPattern,
               ConvertCreateCscOpToGpuRuntimeCallPattern,
               ConvertCreateBsrOpToGpuRuntimeCallPattern,
               ConvertCreate2To4SpMatOpToGpuRuntimeCallPattern,
               ConvertDestroySpMatOpToGpuRuntimeCallPattern,
               ConvertSpMVBufferSizeOpToGpuRuntimeCallPattern,
               ConvertSpMVOpToGpuRuntimeCallPattern,
               ConvertSpMMBufferSizeOpToGpuRuntimeCallPattern,
               ConvertSDDMMBufferSizeOpToGpuRuntimeCallPattern,
               ConvertSpMMOpToGpuRuntimeCallPattern,
               ConvertSDDMMOpToGpuRuntimeCallPattern,
               ConvertSpGEMMCreateDescrOpToGpuRuntimeCallPattern,
               ConvertSpGEMMDestroyDescrOpToGpuRuntimeCallPattern,
               ConvertSpGEMMWorkEstimationOrComputeOpToGpuRuntimeCallPattern,
               ConvertSpGEMMCopyOpToGpuRuntimeCallPattern,
               ConvertSpMatGetSizeOpToGpuRuntimeCallPattern,
               ConvertSetCsrPointersOpToGpuRuntimeCallPattern>(converter);
  patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, gpuBinaryAnnotation, kernelBarePtrCallConv, cachedModuleTable);
  patterns.add<EraseGpuModuleOpPattern>(&converter.getContext());
}
```

For example: `ConvertAllocOpToGpuRuntimeCallPattern`, in the `matchAndRewrite` method of this class, will convert the `gpu.alloc` operation into a method of llvm runtime call.

### `-gpu-module-to-binary`

This pass is used to convert gpu.module into a binary representation. In GPU programming, code is written in a high-level programming language and then converted by a compiler into binary code that the GPU can execute.
This pass is implemented through `struct GPUModuleToBinaryPass` (the code is located at: `lib/Dialect/GPU/Transforms/ModuleToBinary.cpp`).

```c++
void GpuModuleToBinaryPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto targetFormat =
      llvm::StringSwitch<std::optional<CompilationTarget>>(compilationTarget)
          .Cases("offloading", "llvm", CompilationTarget::Offload)
          .Cases("assembly", "isa", CompilationTarget::Assembly)
          .Cases("binary", "bin", CompilationTarget::Binary)
          .Cases("fatbinary", "fatbin", CompilationTarget::Fatbin)
          .Default(std::nullopt);
  if (!targetFormat)
    getOperation()->emitError() << "Invalid format specified.";

  // Lazy symbol table builder callback.
  std::optional<SymbolTable> parentTable;
  auto lazyTableBuilder = [&]() -> SymbolTable * {
    // Build the table if it has not been built.
    if (!parentTable) {
      Operation *table = SymbolTable::getNearestSymbolTable(getOperation());
      // It's up to the target attribute to determine if failing to find a
      // symbol table is an error.
      if (!table)
        return nullptr;
      parentTable = SymbolTable(table);
    }
    return &parentTable.value();
  };

  TargetOptions targetOptions(toolkitPath, linkFiles, cmdOptions, *targetFormat,
                              lazyTableBuilder);
  if (failed(transformGpuModulesToBinaries(
          getOperation(),
          offloadingHandler ? dyn_cast<OffloadingLLVMTranslationAttrInterface>(
                                  offloadingHandler.getValue())
                            : OffloadingLLVMTranslationAttrInterface(nullptr),
          targetOptions)))
    return signalPassFailure();
}
```
The general idea of this code:

1. Create the target format. The target format includes: offloading means that the function will be transferred to the GPU for running, assembly means that the function will be compiled into assembly language, and binary means that the function will be translated into a binary file.

2. Create a callback function lazyTableBuilder for symbol table construction

3. Build a callback function based on the toolkit path, link file, command line option, target format and symbol table to create target options TargetOptions

4. Call the transformGpuModulesToBinaries function to convert the GPU modules into binary representation. This function will traverse all areas of the op, call the moduleSerializer function for each GPUModule in each Block of each Region, and serialize the GPUModule.

```c++
LogicalResult mlir::gpu::transformGpuModulesToBinaries(
    Operation *op, OffloadingLLVMTranslationAttrInterface handler,
    const gpu::TargetOptions &targetOptions) {
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module :
           llvm::make_early_inc_range(block.getOps<GPUModuleOp>()))
        if (failed(moduleSerializer(module, handler, targetOptions)))
          return failure();
  return success();
}
```

The general idea of implementing the moduleSerializer function is as follows:

1. Create an OpBuilder object for building MLIR operations and a SmallVector object for storing serialized objects.

2. Traverse all target attributes of GPUModuleOp. Each target attribute represents a specific GPU target.

3. Use the serializeToObject method of the target object to serialize the GPU module into a binary object, and store the serialized result in serializedModule. Search the MLIR source code. Currently, there are three objects supported by MLIR that can be serialized, namely: NVVM, ROCDL, and SPIRV.

```c++
LogicalResult moduleSerializer(GPUModuleOp op,
                               OffloadingLLVMTranslationAttrInterface handler,
                               const TargetOptions &targetOptions) {
  OpBuilder builder(op->getContext());
  SmallVector<Attribute> objects;
  // Serialize all targets.
  for (auto targetAttr : op.getTargetsAttr()) {
    assert(targetAttr && "Target attribute cannot be null.");
    auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);
    assert(target &&
           "Target attribute doesn't implements `TargetAttrInterface`.");
    std::optional<SmallVector<char, 0>> serializedModule =
        target.serializeToObject(op, targetOptions);
    if (!serializedModule) {
      op.emitError("An error happened while serializing the module.");
      return failure();
    }

    Attribute object = target.createObject(*serializedModule, targetOptions);
    if (!object) {
      op.emitError("An error happened while creating the object.");
      return failure();
    }
    objects.push_back(object);
  }
  builder.setInsertionPointAfter(op);
  builder.create<gpu::BinaryOp>(op.getLoc(), op.getName(), handler,
                                builder.getArrayAttr(objects));
  op->erase();
  return success();
}
```

Taking NVVM as an example, in `class NVVMTargetAttrImpl` (the code is located at: `lib/Target/LLVM/NVVM/Target.cpp`), the serializedModule method is implemented. This method mainly constructs an NVPTXSerializer serializer to implement serialization operations.

```c++
std::optional<SmallVector<char, 0>>
NVVMTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
#if MLIR_CUDA_CONVERSIONS_ENABLED == 1
  NVPTXSerializer serializer(*module, cast<NVVMTargetAttr>(attribute), options);
  serializer.init();
  return serializer.run();
#else
  module->emitError(
      "The `NVPTX` target was not built. Please enable it when building LLVM.");
  return std::nullopt;
#endif // MLIR_CUDA_CONVERSIONS_ENABLED == 1
}
```

In the source code, `NVPTXSerializer` and `SerializeGPUModuleBase` do not override the run method, which means that `serializer.run()` executes the run of the base `class ModuleToObject`. The main idea of the run method is as follows:

1. Call the `translateModuleToLLVMIR` method to convert the module to llvm ir

2. Call loadBitCodeFiles and linkFiles to load and link binary files to llvmmodule

3. Call `moduleToObject` to write the binary data of llvmmodule into a buffer, and call llvm::WriteBitCodeToFile to write the data into the file and return the buffer.

At this point, the conversion between gpu module and binary is completed.

