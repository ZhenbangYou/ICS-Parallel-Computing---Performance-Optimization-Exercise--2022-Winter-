# Why Does GPU Excel in Data-Parallel Applications

## What is Data-Parallel Applications
### Definition
- [Wiki](https://en.wikipedia.org/wiki/Data_parallelism)
- My words: Performing the **same operation** on **different elements**, and these operations are **independent**.
- In other words, there are three key points:
  - SIMD (Single Instruction, Multiple Data)
  - Independence among operations on diffrent data.
### Example
- Scientific computing/HPC (High Performance Computing).
- Computer graphics and image processing.
  - This is why GPU is called GPU.
- Neural network training and inference.
- ~~Cryptocurrency mining.~~

## ILP
- Pipeline: currently 4 stages.
- Multi-issue: currently dual issue.
- Simple pipeline, in-order execution, consuming few transistors and die size. As a result, compared with CPU, more transistor budget and power budget are dedicated to ALU instead of controller.
- Compiler: nvcc.

## DLP
- SIMD: from the perspective of programmers, elements are executed in groups of 32 elements. From the perspective of hardware, in the latest NVIDIA GPU, the number is 16.

## TLP
- Multithreading: fine-grained multithreading. Currently 2048 threads can reside on the same SM (Streaming Multiprocessor), and there can be instructions from different thread blocks in the same pipeline simultaneously. GPUs hide the latency between **Main Memory** and **VRAM** (memory of GPU) with Multithreading.
- Multiprocessing: the core count of GPU is much higher than that of CPU.
  - Comparison between CPU and GPU
    - Processors with most cores (FP32).
      - NVIDIA RTX 3090 Ti (professional graphics card): 10,752.
        - Launch Price: $1,699.
          - However, the real price may be twice or three times as high, due to the shortage of GPU supply.
      - NVIDIA A100 (professional graphics card): 6,912.
      - AMD 3990X: 64 cores, 2 FMA units per core (AVX-2, 256 bits / sizeof(FP32) = 8 elements); that is, $64*2*8=1,024$.
        - Original suggested retail price: $3,999.
      - Intel Xeon Platinum 8380: 40 cores, 2 FMA units per core (AVX-512, 512 bits / sizeof(FP32) = 16 elements); that is, $40*2*16=1,280$.
        - Recommended Customer Price: $8,666.
- Hardware scheduler: CPU threads are scheduled by the OS, which is much more costly.
  - NVIDIA GPU adopts a two-level hardware scheduler.
- Cheap thread context and large register count: consequently, a large number of threads can reside in the register file, so switch among them is extremely cheap -- there is no need to save/restore registers to/from the stack.
## Multiple GPU within a Single Machine
- NVLink and NVSwitch: high-bandwidth bus among GPUs.
## Memory
### Memory Bandwidth
- GDDR and HBM.
  - Due to the nature of data parallelism, the performance depends more on memory BW rather than memory latency.
- Typically, the memory BW of GPU is one order higher than that of CPU.
- Maximal Theoretical Memory BW Comparison
  - CPU
    - DDR4: 8 channels * 3200 MHz * 8 Bytes = 204.8 GB/s.
  - GPU
    - GDDR6X: NVIDIA RTX 3090 Ti: 1,018 GB/s.
    - HBM2e: NVIDIA A100: 2,039 GB/s.
  - Apple Silicon
    - M1 Max (LPDDR5): 400 GB/s.
- **The importance of memory BW is always overlooked.**
### Memory Hierarchy
- Memory hierarchy of GPU is apparently different than that of CPU, e.g., there are software-managed caches in GPU.
- Historically, the cache size of GPU is smaller than that of CPU, but this is turned over by AMD **Infinity Cache** -- on RX 6900XT, there is a 128MB L3 cache.

## Hardware-Software Codesign
- CUDA, including highly optimized libraries like cuDNN, cuBLAS, cuFFT, etc.
  - CUDA is faster than OpenCL.