# BactoML - Bactosense Machine Learning libraries

* Marie Sadler, Octave Martin, Eliane Roosli, Ilaria Ricchi, Douglas Watson
* bNovate SA, 2018-2020

## Overview

This python package contains all of our custom machine learning tools. They are built for maximum interoperability with Scikit-learn, so the APIs should be familiar to most. We provide several modules related to fingerprinting:

- **Data loading**: loading of FCS files
- **Data generation**: generation of FCS files with artificial contaminations
- **Pre-processing**: tlog transforms, gating to remove noise
- **Fingerprinting**: gating to count cells, probability binning
- **Estimators**: Chi-square risk score estimator.

## Getting started

Create a virtual environment. I recommend using conda to ensure you have the correct python version:

```
conda create --name bactoml python=3.6 --no-default-packages
conda activate bactoml
```

**Note**: You must run the last line (`conda activate bactoml`) every time you open a new shell to work on this package!

Install the package with:

```
pip install -r requirements.txt
python setup.py install
```

[We use pip inside of conda, because we can't get all packages through conda, so it's easier to use only pip]

Then work through the examples in the `examples/` folder.

For developers: again I recommend working in a virtual environment or Anaconda environment. Install the library in development mode in your virtualenv:

```
pip install -r requirements-test.txt
python setup.py install
python setup.py develop
```

Then create a branch and hack away. 

### Testing

Tests use pytest. Install py.test:
```
pip install pytest
```

And run:
```
pytest
```

### Further documentation

Documentation is in the docs/ folder, and is managed by sphinx. To build the documentation, make sure you have installed the developer requirements, then:
```
cd docs/
make html
```

The output is in `docs/html/_build/index.html`

Document your code in numpydoc format. When working on the sphinx rst files, you may find it useful to automatically build the docs every time you save a file:
```
./scripts/auto_build_doc
```


## Contribution guidelines

See CONTRIBUTING.md.