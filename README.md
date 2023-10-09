This repository contains the source code of the ML quantum state tomography experiment in the paper "Fast Minimization of Expected Logarithmic Loss via Stochastic Dual Averaging" submitted to AISTATS 2024.

# How to run
- Tested on [Julia](https://julialang.org) Version 1.9.2
- Set the dimension and the number of samples in `settings.jl`
## Install Packages
```
$ cd mlqst/
$ julia ./install.jl
```
## Run
```
$ julia ./main.jl
```

# Implemented Algorithms
## Batch Algorithms
1. Iterative MLE: A. I. Lvovsky, Iterative maximum-likelihood reconstruction in quantum homodyne tomography, *J. opt., B Quantum semiclass. opt.*, 2004 ([link](https://arxiv.org/abs/quant-ph/0311097))
2. QEM: Chien-Ming Lin, Hao-Chung Cheng, and Yen-Huan Li, Maximum-likelihood quantum state tomography by Cover's method with non-asymptotic analysis, *arXiv preprint*, 2022 ([link](https://arxiv.org/abs/2110.00747))
3. Monotonous Frank-Wolfe: A. Carderera, M. Besançon, and S. Pokutta, Simple steps are all you need: Frank-Wolfe and generalized self-concordant functions, *Adv. Neural Information Processing Systems*, 2021 ([link](https://proceedings.neurips.cc/paper/2021/hash/2b323d6eb28422cef49b266557dd31ad-Abstract.html))
4. Bregman proximal gradient: Bauschke, Jérôme Bolte, Marc Teboulle, A Descent Lemma Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited and Applications, *Math. Oper. Res.*, 2017 ([link](https://pubsonline.informs.org/doi/abs/10.1287/moor.2016.0817))
5. Frank-Wolfe: Renbo Zhao and Robert M. Freund, Analysis of the Frank–Wolfe method for convex composite optimization involving a logarithmically-homogeneous barrier, *Math. Program.*, 2023 ([link](https://link.springer.com/article/10.1007/s10107-022-01820-9))
6. Entropic mirror descent with Armijo line search: Yen-Huan Li and Volkan Cevher, Convergence of the Exponentiated Gradient Method with Armijo Line Search, *J. Optim. Theory Appl.*, 2019 ([link](https://link.springer.com/article/10.1007/s10957-018-1428-9))
7. Diluted iMLE with Armijo line search: D. S. Gonçalves, M. A. Gomes-Ruggiero, and C. Lavor, Global convergence of diluted iterations in maximum-likelihood quantum tomography, *Quantum Inf. Comput.*, 2014 ([link](https://arxiv.org/abs/1306.3057))
## Stochastic Algorithms
1. Stochastic Q-Soft-Bayes: Chien-Ming Lin, Yu-Ming Hsu, and Yen-Huan Li, Maximum-likelihood quantum state tomography by Soft-Bayes, *arXiv preprint*, 2022 ([link](https://arxiv.org/abs/2012.15498))
2. Stochastic Q-LB-OMD: Chung-En Tsai, Hao-Chung Cheng, and Yen-Huan Li, Faster stochastic first-order method for maximum-likelihood quantum state tomography, *arXiv preprint*, 2022 ([link](https://arxiv.org/abs/2211.12880))
3. 1-sample LB-SDA: **this work**
4. $d$-sample LB-SDA: **this work