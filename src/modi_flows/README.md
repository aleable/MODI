<p align="center">
<a href=https://en.wikipedia.org/wiki/Amedeo_Modigliani><img src="https://user-images.githubusercontent.com/34717973/163191831-69d0a9d0-eadd-4bf4-bc65-836f2cda5fcb.png" width="650"></a>
</p>

## Implementation details

### Table of Contents  

- [What's included](#whats-included)  
- [How to use](#how-to-use)  
    - [Parameters](#parameters)  
- [I/O format](#io-format)  
    - [Input](#input)
    - [Output](#output)
- [Additional implementation details](#additional-implementation-details)
    - [Transport network setup](#transport-network-setup)
    - [Convergence criteria](#convergence-criteria)
- [Usage examples](#usage-examples)  
- [Contacts](#contacts)
- [License](#license)


## What's included

- ```dashboard.ipynb```: Jupyter notebook containing an easy-to-use interface with MODI
- ```utils.ipynb```: auxiliary functions needed by ```dashboard.ipynb```
- ```main.py```: main function containing the MODI class, and a ```preprocessing()``` function to build the problem, given a pair of images
- ```initialization.py```: construction of the transport network and of the multicommodity problem
- ```dynamics.py```: finite difference scheme to solve the multicommodity dynamics


## How to use

### Parameters

The parameters you can pass to MODI are introduced in ```dashboard.ipynb```. They are:

- *Problem construction*
    - ```otp_alg``` = ```"multicom"``` or ```"unicom"```: choose how many commodities you want to run the dynamics with. Set ```multicom``` for M = 3, i.e., to use colored images, and M = 1 to use grayscale ones
    - ```sigma``` (```type=float```): standard deviation for an additional Gaussian smoothing of the images (if needed)
    - ```pooling_size``` (```type=int```): kernel size for an additional average pooling of the images (if needed)
    - ```theta``` (```type=float```): 0 ≤ θ ≤ 1, convex combination weight between pixels' positions and colors contribution to the ground cost
    - ```alpha``` (```type=float```): α ≥ 1/2 (suggested), penalty for Kirchhoff's law relaxation

- *Dynamics parameters*
    - ```time_step``` (```type=float```): Δt > 0, time step for the finite difference discretization of the dynamics
    - ```time_tol``` (```type=int```): upper bound on the number of time steps (safety variable)
    - ```tol``` (```type=float```):  threshold for convergence of the multicommodity dynamics
    - ```beta``` (```type=float```): 0 < β < 2, regulatory parameter for transport paths consolidation

- *Misc*
    - ```VERBOSE``` (```type=bool```): print info while running the scheme

## I/O format

### Input

If you want to test the code you need to pass as input files images as ```np.arrays```, with the standard formatting used for RGB data. Particularly, an ```image``` variable should be an array with shape ```(w,h,3)``` = (width, height, color channels), and with all its entries converted to integers in the range [0, 255].

To faciliate the understanding of this preprocessing step, we include the function ```utils.preprocessing()``` in ```dashboard.ipynb```, where a subset of [Jena Flowers](https://doi.org/10.7910/DVN/QDHYST) is converted to the desired format.
### Output

The outputs returned by our scheme are:

- ```j```: optimal transport cost at convergence
- ```self.r```: tensor H, transported in the multicommodity problem
- ```self.s```: tensor G, transported in the multicommodity problem
- ```self.c```: ground cost matrix used to build the transport network

These can be serialized using the functions ```exec()``` in ```main.py```. In case one wants to export only the three latter variables without running the multicommodity dynamics, the function ```setup_only()``` in ```main.py```can be utilized.


## Additional implementation details

### Transport network setup

The multicommodity dynamics is executed on the transport network K, which is build in ```initialization.py```. In detail, its construction is made of the following steps:

```python
self.c, self.r, self.s = sparsunb(*args)    # 1.
edges = index_edges(*args)                  # 2a.
length = cost_edge(*args)                   # 2b.
self.B, self.length = topology(*args)       # 2c.
self.forcing = rhs_construction(*args)      # 3.
```

1. given a complete bipartite nertwork between a pair of images, we sparsify it using the trimming proposed in [1]. Moreover, we extend the optimal transport formulation to account for G and H with different total mass
2. (a) we extract the list of edges of K, and (b-c) its incidence matrix. In these last two steps, we also assign a cost to each link
3. we construct the forcing S from G and H

[1]  Ofir Pele, Michael Werman, [ECCV 2008 (2017)](https://ieeexplore.ieee.org/document/5459199).

### Convergence criteria

The convergence criteria we choose for our algorithms is the following. We stop the code when the cost difference between two consecutive time steps is below the threshold ```tol```. Schematically:
```python
dc = abs(cost_update - cost)/self.time_step
if dc < self.tol and it > 5:    # a lower bound on the number of iteration (it) is added for safety
    conv = True
```


## Usage examples

For a basic usage example of the code you can simply take a look at ```dashboard.ipynb```. <br/>
The execution of MODI is performed in two steps:
- *Problem setup*, i.e. construction of the transport network given a pair of images:

```python
from main import *  # import preprocessing scripts
# [...] you may want to load other packages

otp_alg = "multicom" # "multicom"/"unicom" for M=3 and M=1, respectively
sigma = 0.5          # in case additional gaussian smoothing of the images is desired
pooling_size = 1     # in case further pooling of the images is desired
theta = 0.5          # convex weight for the construction of C
alpha = 0.5          # penalty for Kirchhoff's law (https://ieeexplore.ieee.org/document/5459199)


# select two samples, here taken from the Jena Flowers 30 Dataset
img1 = flowers_dataframe["image"][0]
img2 = flowers_dataframe["image"][1]

# serialize the ground cost, and the transport tensors G, H
C, g, h = preprocessing(img2, img1, otp_alg, pooling_size, sigma, theta)
thresh = 0.05  # trimming threshold (https://ieeexplore.ieee.org/document/5459199)
```
- *Execution*, you just need to run:
```python
from main import MODI # import multicommodity dynamics class

# parameters used to execute the multicommodity dynamics
time_step = 0.5  # forward Euler discretization time step
tol = 1e-1       # convergence tolerance
time_tol = 1e3   # stopping time step limit (for safety)
beta = 1.0       # regularization parameter

modi = MODI(g, h, C,
            beta=beta,
            dt=time_step,
            tol=tol,
            time_tol=time_tol,
            alpha=alpha,
            t=thresh,
            verbose=VERBOSE)

ouputs = modi.exec() # as described in in Sec. I/O Format
```

## Contacts

For any issues or questions, feel free to contact us sending an email to:
- <a href="alessandro.lonardi@tuebingen.mpg.de">alessandro.lonardi@tuebingen.mpg.de</a><br/>
or
- <a href="diego.theuerkauf@tuebingen.mpg.de">diego.theuerkauf@tuebingen.mpg.de</a>

## License

Copyright (c) 2022 <a href="https://aleable.github.io/">Alessandro Lonardi</a>, <a href="https://github.com/diegoabt">Diego Baptista</a> and <a href="https://www.cdebacco.com/">Caterina De Bacco</a>

<sub>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:</sub>

<sub>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.</sub>

<sub>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</sub>
