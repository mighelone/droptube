# Droptube reactor

Char gasification example in a drop tube reactor.
The equations necessary for calculating the char evolution in a drop tube reactor are coded in the `python` module `char.py`.

A drop tube reactor example is presented in the notebook `droptube.ipynb`.

## Installation and requirements

The notebook and the module work with `python 3.5`. It has not been tested with older version of `python`.
The following additional modules are required:

- `numpy`
- `matplotlib`
- `scipy`

### Install modules using PIP

The missing modules can be installed, if they are not yet present using `pip`

```
# pip install --local numpy
# pip install --local matplotlib
# pip install --local scipy
# pip install --local jupyter
```

The option `--local` allow to install locally the additional packages.

## Run notebook

The notebook can be run, using the following command:

```
# jupyter-notebook droptube.ipynb
```

The whole notebook can be ran with the command `Cell`-`Run All`.
