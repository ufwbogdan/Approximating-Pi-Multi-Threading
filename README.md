Approximating Pi by means of numerical integration using Multi Threading (OpenMP) in C++.

  - Given that LaTeX: \pi  is the area of a circle with unit radius, we approximate LaTeX: \pi  by computing the area of a quarter circle using Riemann sums as described in the following method.
  - Let ```math f\left(x\right)=\sqrt{1-x^2} ``` be the formula describing the quarter circle for ```math x\in\left[0,1\right]\subseteq\mathbb{R} ```. We approximate the quarter circle area by means of the formula:
  ```math
  \frac{\pi}{4} \approx \sum_{i=0}^{N-1} \Delta x \, f(x_i)
  ```
  where ```math x_i = i \Delta x and \Delta x = \frac{\1}{N}```. Such an approximation becomes more accurate as N approaches to infinity.

  - Other than the parallel implementation exploiting multi-threading, i.e. OpenMP, there is another unse that makes use of both multi-threading and vectorization, i.e. OpenMP and intel instrinsic functions.
