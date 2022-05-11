# Guideline of DGEMM optimization

## Preliminary
The optimziation of DGEMM (Double GEneral Matrix Multiplication) is a classic problem in HPC (High Performance Computing). And this is discussed in depth in *Computer Organization and Design: The Hardware-Software Interface*.

## Compilation
```g++ -o dgemm dgemm.cpp -O2 -std=c++17"```

## Understand the Code
The following is some questions that may help you understanding.
1. Why *\<cmath\>* rather than *\<math.h\>*?
2. What is the usage of *constexpr* (as a specifier of variables, arguments, and functions) versus *const*?
3. Why *kAlignInBytes*, *kAlignInElements*, and *std::align_val_t*? What is the role of **alignment** in *AVX*?
4. If *kRepetitions* is set to *1*, how unstable will the outcome be?
5. What does the initialization *ElementType sum {}* mean? Note that this is a variable of a basic type (rather than a user-defined class).
6. *std::rand* is used as the random number generator here. Find out its disadvantages. Also find something about the new random number generator provided by **C++11** [here](https://en.cppreference.com/w/cpp/numeric/random).
7. Get some hands-on experiences with **C++ clock**. The reference can be found [here](https://en.cppreference.com/w/cpp/chrono#Clocks). *chrono* is extremely powerful! Find more about it!
8. What is the usage of *std::tuple*? Read [this](https://en.cppreference.com/w/cpp/utility/tuple).
9. This code snippet is cross-platform. Compared with **C**, what features of **C++** is utilized to realize this?

## Optimization Guide
1. Memory locality
2. ILP (Instruction-Level Parallelism)
  - Cooperate with the compiler! [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
3. DLP (Data-Level Parallelism)
  - [AVX](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html). Besides, what is the motivation of **FMA**?
    - The slide about how to write **AVX intrinsics** is also uploaded.
    - To practice **AVX intrinsics**, start with DAXPY (Double $\alpha$ X + Y).
  - For ARM, refer to the **Recommended Computer System Learning Materials** I wrote.
4. TLP (Thread-Level Parallelism)
  - Try **C++ thread/jthread**
  - **OpenMP**. *#pragma omp parallel for*. Compile with *-fopenmp*. More information should be figured out by yourself.