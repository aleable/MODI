<p align="center">
<a href=https://en.wikipedia.org/wiki/Amedeo_Modigliani><img src="https://user-images.githubusercontent.com/34717973/163191831-69d0a9d0-eadd-4bf4-bc65-836f2cda5fcb.png" width="650"></a>
</p>

___
<h1 align="center">
MODI: Multicommodity Optimal Transport Dynamics on Images
</h1>
<p align="center">
<a href="https://arxiv.org/abs/2205.02938" target="_blank">
<img alt="ARXIV: 2205.02938" src="https://img.shields.io/badge/arXiv-2212.08593-red.svg">
</a>
  
<a href="https://www.treedom.net/en/page/register?id=49Z-KEWX" target="_blank">
<img alt="Treedom" src="https://img.shields.io/badge/CO2%20compensation%20-Treedom%20%F0%9F%8C%B4-brightgreen">
</a>

<a href="https://pypi.org/project/modi-flows/" target="_blank">
  <img alt="PyPI Version" src="https://img.shields.io/pypi/v/modi-flows">
</a>

</p>

> <strong>&#9888; Important note:<br/></strong> MODI is currently under reconstruction, thus you may find some inconsistencies in its documentation. In case you have problems using the code, please do not hesitate to contact us.


**MODI** (**M**ulticommodity **O**ptimal transport **D**ynamics on **I**mages) is a Python implementation of the algorithms used in:

- [1] Alessandro Lonardi\*, Diego Baptista\*, and Caterina De Bacco. <i>Immiscible Color Flows in Optimal Transport Networks for Image Classification</i>. <a href="https://doi.org/10.3389/fphy.2023.1089114">Front. Phys. 11:1089114</a> [<a href="https://arxiv.org/abs/2205.02938">arXiv</a>] [<a href="https://github.com/aleable/MODI/tree/main/misc/POSTER_MODI.pdf">poster</a>] [<a href="https://www.treedom.net/en/page/register?id=49Z-KEWX">CO₂ compensation</a>].

This is a scheme capable of performing supervised classification by finding multicommodity optimal transport paths between a pair of images.

**If you use this code, please cite [1].**<br/>
The symbol “*” denotes equal contribution.


## Requirements

All the dependencies needed to run the algorithm can be installed using the following command:

```bash
pip install modi-flows
```

Please note that as of the latest release, the `scikit-umfpack` package is no longer a mandatory requirement for `modi-flows`. However, we highly recommend installing it to take advantage of enhanced performance. If you choose to install `scikit-umfpack`, it can be easily obtained from the conda repository:

```bash
conda install -c conda-forge scikit-umfpack
```
Now, you are ready to use the code! To do so, you can simply use the notebook ```dashboard.ipynb```, from which you can access our solver. <br/>


Sure, here's the updated section for the directory structure:

## What's included

- `code`: Contains all the scripts necessary to run MODI, including the main implementation in `src/modi_flows/`.
- `notebooks`: Holds user-friendly Jupyter notebooks, such as `dashboard.ipynb`, which allow you to interact with the code and visualize results.
- `data`: Contains input data used in the examples.
  - `input`: Holds a small sample of images taken from [2]. These images can be preprocessed using `code/dashboard.ipynb`. The complete dataset can be downloaded as a .zip file from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QDHYST).
- `misc`: Includes supplementary files such as the MODI poster.
- `tests`: Contains test scripts to validate the functionality of the code.
- `docs`: Contains documentation files, including Sphinx configuration and source files.

[2]  Marco Seeland, Michael Rzanny, Nedal Alaqraa, Jana Wäldchen, and Patrick Mäder, [Jena Flowers 30 Dataset, Harvard Dataverse (2017)](https://doi.org/10.7910/DVN/QDHYST).


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
