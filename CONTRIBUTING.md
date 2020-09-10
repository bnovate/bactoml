# Contributing to BactoML


* Follow the PEP8 guidelines for Python code style (use a Linter)
* Follow scikit learn's API: http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects
* Document all function, classes, and modules with docstrings.
    * Add any new module to the :autodoc: statements in the sphinx docs.
* As much as possible, write unit tests for new modules, see examples in `bactoml/tests` 
* Illustrate new modules by writing a tutorial in `tutorial`. We avoid jupyter notebooks in that folder, as they tend to create git conflicts.