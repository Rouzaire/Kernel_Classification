Ylann Rouzaire, all rights reserved.

# Kernel Classification

## Goal of the project

Investigate the dynamics of learning of a hard-margin 2-class Kernel Classification task.

## Structure of the code

The project is small enough for the code to be organized as follows :
* All the functions are defined in `function_definitions.jl`
* The `main.jl` defines the keys arguments, among them the number of statistics to collect and distributes the work on different processors thanks to the function `Run`. The user is free to choose, depending on its machine and the desired input parameters, whether to parallelize with respect to the gap or with respect to the dimensions.
    * The function `Run` calls the correct function depending on what the user wants to parallelize wrt : `Run_fixed_delta` if wrt gap or `Run_fixed_dimension` if wrt dimension
* The data is saved in JLD files by these 2 functions so that the analysis can be performed later by the `analysis.jl` file.
* The `benchmark.jl` is a test file and _may contain deprecated syntax_.
* The `decision_boundary.jl` aims at investigating the effect of a gap on the decision function (f) and thus on the prediction (sign of f). Note that resulting pdf figures can be quite heavy (~10Mo) when using moderatly fine grids.

## Generation of the data

The data is generated uniformly on the unit hypersphere of dimension d (for clarity : d=1 means the unit circle and d=2 means the usual sphere embedded in the natural 3 dimensions) by normalizing (to 1) points from a multivariate random normal distribution in d+1 dimensions.

In case of a finite gap Δ, there must be no data such that |x1| < Δ. The strategy is simple : generate more samples and throw away samples such that |x1| < Δ.

Note that the labeling is internally handled by the `generate_Y` function.
* `generate_TrainSet` has nothing special, data is uniformly distributed on the unit hypersphere \ the gap Δ
* On the other hand, `generate_TestSet` is slightly different. Because when classifying with a gap between interfaces, the probability to misclassify is much smaller (in fact, the learning curves decrease exponentially), the size of the trainset grows exponentially fast and quickly saturates the RAM. Therefore, I implemented sort of an **Importance Sampling** (IS) algorithm (at least conceptually). The idea is that if a test point is far from the interface, it will for sure be correctly classified : it is therefore useless to compute that prediction. `generate_TestSet` thus only generates data points in the vicinity of the gap. The choice of the distance to the gap is of capital importance. It must be sufficiently small for the IS to be effective, but it must be sufficiently large to catch all possibly misclassified points, otherwise the estimate would be very incorrect. That distance is encoded in the variable `SVband`. The `generate_TestSet` also returns the IS weight called `weight_band`, which is simply the probability to fall into that 'SVband' region.


## Some remarks
* The kernels we use are from the Matérn family : *[Read more](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)*. These kernels are isotropic and translation invariant. They are coded in the function `Matérn(h)`. Note that the Laplace kernel corresponds to Matérn[ν = 1/2].

* The actual classification is taken care of by the *[SVC algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)* of the Python *sklearn* library. This library is called through Julia thanks to the *[PyCall Julia package](https://github.com/JuliaPy/PyCall.jl/)*. This library exports a soft-margin algorithm, which can emulate a hard-margin SVM if the cost C >> 1 (set to 1E10 in the code), completely prohibiting misclassification during training.
     * The charges (α dual coefficients of the support vectors) are delivered by the `dual_coef_` attribute of the Python classifier. Note that they are signed.
     * The distances between nearest support vectors is not supported by any native attribute/method : the functions `rc` and `compute_rc` handle these calculations.     

* The code is designed to collect statistics by running M independent realisations. 20 - 40 realisations should be sufficient for smooth curves and small standard deviation for most of problems.

* The leading term in runtime complexity is O(#dimensions * #gaps * P^2 * M). If using the default Gaussian kernel of the classifier, the code is about 100x faster than if using a precomputed kernel because the precomputation of the Gram Matrix handled by `Kernel_Matrix` must be less optimized (for instance, the entire matrix is created and stored in memory, there must be a more clever way).




## Bibliography
* Paccolat, Spigler and Wyart : How isotropic kernels learn simple invariants
