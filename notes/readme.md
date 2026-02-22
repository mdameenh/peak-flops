
# Study plan

## Month 1: GPU Architecture, C++, and CUDA Foundations

### Week 1: GPU Architecture & Memory Hierarchy
- Study GPU architecture: parallelism, memory hierarchy, SIMD/SIMT, CUDA cores ([NVIDIA Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/), [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html))
- Compare NVIDIA architectures (Turing, Ampere, Ada Lovelace)
- Watch GTC talks on GPU architecture ([NVIDIA GTC On-Demand](https://www.nvidia.com/en-us/gtc/on-demand/))

### Week 2: CUDA Programming Basics
- Install CUDA Toolkit and set up a development environment ([CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads))
- Complete CUDA C/C++ programming tutorials ([CUDA Getting Started Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html))
- Implement simple vector addition and matrix multiplication ([CUDA Samples](https://github.com/NVIDIA/cuda-samples))

### Week 3: CUDA Kernels, Streams, and Profiling
- Learn about CUDA kernels, streams, and events ([CUDA Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams))
- Use CUDA profiling tools (Nsight, Visual Profiler) ([NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-visual-studio-edition))
- Practice optimizing memory access and kernel launches ([CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html))

### Week 4: Modern C++ for CUDA Programmers

**Day 1: C++ Templates (Functions & Classes)**  
- Why templates matter in CUDA (generic kernels, type flexibility)  
	[C++ Function Templates](https://en.cppreference.com/w/cpp/language/function_template)  
	[C++ Class Templates](https://en.cppreference.com/w/cpp/language/class_template)

**Day 2: Lambda Expressions & Functors**  
- Using lambdas for device code and kernel launches  
	[C++ Lambda Expressions](https://en.cppreference.com/w/cpp/language/lambda)  
	[Functors in C++](https://en.cppreference.com/w/cpp/utility/functional)

**Day 3: Smart Pointers & Resource Management**  
- When to use (and not use) smart pointers in CUDA host code  
	[std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr)  
	[std::shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr)  
	[RAII Principle](https://en.cppreference.com/w/cpp/language/raii)

**Day 4: Move Semantics & Rvalue References**  
- Efficient memory transfers and avoiding unnecessary copies  
	[Move Semantics](https://en.cppreference.com/w/cpp/language/move_constructor)  
	[Rvalue References](https://en.cppreference.com/w/cpp/language/reference)

**Day 5: Standard Library Algorithms & Iterators**  
- Using STL algorithms for host-side data prep and validation  
	[std::vector](https://en.cppreference.com/w/cpp/container/vector)  
	[std::algorithm](https://en.cppreference.com/w/cpp/algorithm)  
	[Iterators](https://en.cppreference.com/w/cpp/iterator)

**Day 6: Error Handling & Exceptions**  
- Error handling in C++ vs. CUDA (no exceptions on device)  
	[C++ Exception Handling](https://en.cppreference.com/w/cpp/language/exceptions)  
	[CUDA Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)

**Day 7: Review & Mini-Project**  
- Review the week’s topics  
- Implement a templated vector addition on host and device (using C++ templates and CUDA)  
	[CUDA C++ Templates Example](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/template/template.cu)


---

## Month 2: Deep Learning, Tensor Operations, and Optimization

### Week 5: Tensor Operations & cuBLAS/cuDNN
- Study tensor operations and linear algebra on GPU ([cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cudnn))
- Implement and benchmark matrix multiplication, convolutions, and reductions

### Week 6: Deep Learning on NVIDIA Hardware
- Study deep learning frameworks and GPU acceleration ([TensorRT](https://developer.nvidia.com/tensorrt), [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples))
- Run and optimize a neural network on GPU
- Explore NVIDIA’s DeepStream SDK for perception ([DeepStream SDK](https://developer.nvidia.com/deepstream-sdk))

### Week 7: Performance Optimization & Best Practices
- Profile and optimize CUDA code ([Nsight Compute](https://developer.nvidia.com/nsight-compute))
- Study memory optimization (shared, global, constant memory) ([CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy))
- Implement and optimize a real-world algorithm (e.g., image processing)

### Week 8: Tooling for Embedded & Real-Time Systems
- Review RTOS concepts (FreeRTOS, QNX, VxWorks) ([FreeRTOS](https://www.freertos.org/), [QNX](https://blackberry.qnx.com/en), [VxWorks](https://www.windriver.com/products/vxworks))
- Study NVIDIA’s embedded platforms (Jetson, Xavier) ([NVIDIA Jetson](https://developer.nvidia.com/embedded-computing))
- Build and deploy a simple application on Jetson Nano ([Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit))

---

## Month 3: Industry, Trends, and Interview Preparation

### Week 9: NVIDIA Ecosystem & Industry Trends
- Research NVIDIA’s business units, products, and recent innovations ([NVIDIA Official Site](https://www.nvidia.com/en-us/about-nvidia/))
- Study the European automotive and GPU market landscape ([Automotive News Europe](https://europe.autonews.com/), [Statista Automotive](https://www.statista.com/markets/418/topic/482/automotive/))
- Read NVIDIA’s annual reports, blogs, and press releases ([NVIDIA Investor Relations](https://investor.nvidia.com/), [NVIDIA Blog](https://blogs.nvidia.com/))

### Week 10: Automotive & Robotics Platforms
- Study AUTOSAR, ISO 26262, and automotive safety standards ([AUTOSAR](https://www.autosar.org/), [ISO 26262 Overview](https://www.iso.org/standard/43464.html))
- Explore NVIDIA DRIVE platform, DRIVE OS, and DRIVE AGX ([NVIDIA DRIVE Platform](https://developer.nvidia.com/drive))
- Explore NVIDIA SDKs relevant to your interest ([Isaac SDK](https://developer.nvidia.com/isaac-sdk), [DRIVE SDK](https://developer.nvidia.com/drive))

### Week 11: System Design, C++/CUDA Problem Solving
- Practice system design interviews ([System Design Primer](https://github.com/donnemartin/system-design-primer))
- Solve LeetCode/HackerRank problems ([LeetCode](https://leetcode.com/), [HackerRank](https://www.hackerrank.com/domains/tutorials/10-days-of-javascript))
- Experiment with sample applications and modify them ([NVIDIA Developer Samples](https://developer.nvidia.com/samples))

### Week 12: Application, Networking, and Interview Prep
- Update your CV and LinkedIn ([NVIDIA Careers](https://www.nvidia.com/en-us/about-nvidia/careers/))
- Prepare STAR stories for behavioral interviews ([STAR Method](https://www.themuse.com/advice/star-interview-method))
- Practice technical interviews ([Interviewing.io](https://interviewing.io/), [Pramp](https://www.pramp.com/))
- Apply to relevant NVIDIA roles and network with current employees ([NVIDIA LinkedIn](https://www.linkedin.com/company/nvidia/))

---