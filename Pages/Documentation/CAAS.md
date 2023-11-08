---
layout: default
title: Buddy CAAS
parent: Documentation
---

## Buddy Compiler As A Service (Buddy-CAAS)

![Buddy-CAAS](../../Images/buddy-caas.png)

### ABSTRACT

MLIR is an extensible and reusable multi-layer intermediate representation
(IR) infrastructure, and its multi-level dialect structure and
mixability make it very expressive and easy to optimize [1]. It can
also represent the abstraction of hardware backend at a high level
to accomplish specific optimizations. However, the cost of these features
is that the compiler’s pass pipeline and toolchain are difficult
to configure. Compiler developers mostly use the command line or
scripts to debug conversions between multiple dialects to find the
optimal pass combination, which is a very time-consuming process.
In order to improve the efficiency of compiler developers, we implement
an online MLIR development aid platform called Buddy-CAAS
(Buddy Compiler As A Service, https://buddy.isrc.ac.cn/). Our platform
allows for fast and flexible configuration and debugging of the
pass pipeline, as well as the visual presentation of the intermediate
product. Furthermore, we can also integrate the toolchain for specific
hardware backend, which can save developers much time in
configuring complex cross-compilation environments. In addition,
we provide some MLIR examples to help users write MLIR code
faster, and users can also share the code with a link in a collaboration
scenario. Our platform has been used many times in discussions
with upstream to help us show MLIR examples. Currently, Buddy-
CAAS is maintained by the Buddy Compiler community, and we are
integrating more features to power the MLIR ecosystem.

### PASS PIPELINE CONFIGURATION

Buddy-CAAS provides a fast and flexible approach to configure
MLIR pass pipeline, which can significantly reduce
developers’ debugging time. MLIR developers often need enough
experience to configure a pass pipeline for a correct lowering. If
targeting high-performance code generation, developers also need
to know the feature of each pass and try different passes repeatedly.
This process is very time-consuming in practice, and there are no
good aid tools. As a result, developers will switch back and forth between
the command line and the editor to find a proper pass pipeline.
To solve this problem, we have integrated the pass pipeline configuration
tool (Config-Picker, Figure 1-b) in Buddy-CAAS. Users can
search for a suitable pass with keyword, add it to the pass pipeline,
and change the pass position by dragging and dropping (Figure 1-d).
Once the entire pipeline is configured, users can also disable some
of the passes to achieve step debugging (Figure 1-c).

### TOOLCHAIN INTEGRATION

Buddy-CAAS integrates with hardware toolchains and emulators,
which can significantly save developers time in building
environment. MLIR is able to provide hardware-specific abstractions
and code generation for different hardware backends via LLVM.
Building backend toolchains and emulators is experienced and timeconsuming,
especially when it comes to cross-compiling. Buddy-
CAAS allows users to use the pre-built tools directly, and users can
see the intermediate products of each compilation stage as well as
the final execution results (Figure 1-e).We hope this feature can help
developers quickly verify the MLIR code on specific backend without
the hassle of environment building process. Currently, we mainly
support RISC-V Vector (RVV) extension toolchain [2] and QEMU
emulator, which addresses the pain point for many RVV-oriented
developers due to the lack of actual hardware. We are also working
on integrate more toolchains and emulators in our platform.

### REFERENCES

[1] Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques
Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko.
Mlir: Scaling compiler infrastructure for domain specific computation. In 2021
IEEE/ACM International Symposium on Code Generation and Optimization (CGO),
pages 2–14. IEEE, 2021.

[2] riscv-v-spec contributors. Risc-v "v" vector extension. https://github.com/riscv/
riscv-v-spec/releases/tag/v1.0.
