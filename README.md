<p align="center">
<a href=https://en.wikipedia.org/wiki/Amedeo_Modigliani><img src="https://user-images.githubusercontent.com/34717973/163191831-69d0a9d0-eadd-4bf4-bc65-836f2cda5fcb.png" width="650"></a>
</p>

___

**MODI** (**M**ulticommodity **O**ptimal transport **D**ynamics on **I**mages) is a Python implementation of the algorithms used in:

- [1] Alessandro Lonardi\*, Diego Baptista\*, and Caterina De Bacco. <i>Immiscible Color Flows in Optimal Transport Networks for Image Classification</i> [<a href="https://arxiv.org/abs/2205.02938">arXiv</a>].

This is a scheme capable of performing supervised classification by finding multicommodty optimal transport paths between a pair of images.

**If you use this code please cite [1].**<br/>
The symbol “*” denotes equal contribution.


## What's included

- ```code```: contains the all the scripts necessary to run MODI, and a user-friendly Jupyter notebook (```dashboard.ipynb```) to interact with the code and visualize the results
- ```data/input```: contains a small sample of images taken from [2], these can be preprocessed using ```code/dashboard.ipynb```. The original dataset can be directly downloaded as a .zip file [from the Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QDHYST)
- ```setup.py```: setup file to build the Python environment

[2]  Marco Seeland, Michael Rzanny, Nedal Alaqraa, Jana Wäldchen, and Patrick Mäder, [Jena Flowers 30 Dataset, Harvard Dataverse (2017)](https://doi.org/10.7910/DVN/QDHYST).

## How to use

To download this repository, copy and paste the following:

```bash
git clone https://github.com/aleable/MODI
```


**You are ready to test the code! But if you want to know how click [HERE](https://github.com/aleable/MODI/tree/main/code).**

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
