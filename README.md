Approximating Pi by means of numerical integration using Multi Threading (OpenMP) in C++.

- Given that pi is the area of a circle with unit radius, we approximate pi by computing the area of a quarter circle using Riemann sumsas described in the following method.

- Let
```math
f(x) = \sqrt{1 - x^2} \text{ for } x \in [0,1] \subseteq \mathbb{R}.
```
We approximate the quarter circle area using:

```math
\frac{\pi}{4} \approx \sum_{i=0}^{N-1} \Delta x \, f(x_i)
```

where 
```math 
x_i = i \Delta x \text{ and } \Delta x = \frac{1}{N} .
```  
The approximation becomes more accurate as N approaches to infinity.

- Besides the parallel OpenMP implementation, there is also a version combining multi-threading and SIMD vectorization using OpenMP and Intel intrinsic functions.
