# ark-spatial

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/angelolab/ark-spatial/test.yaml?branch=main
[//]: # "[link-tests]: .github//workflows/test.yml"
[badge-docs]: https://img.shields.io/readthedocs/ark-spatial

An early development implementation of [`angelolab/ark-analysis`](https://github.com/angelolab/ark-analysis) using [`scverse/spatialdata`](https://github.com/scverse/spatialdata).

## Getting started

Please refer to the [documentation](https://ark-spatial.readthedocs.io/). In particular, the

-   [API documentation](https://ark-spatial.readthedocs.io/en/latest/api.html).
-   [Notebooks](https://ark-spatial.readthedocs.io/en/latest/notebooks.html).

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).
There are a few ways to install `ark-spatial`.

1. Install the latest development version:
    ```shell
    pip install git+https://github.com/angelolab/ark-spatial.git@main
    ```
2. Install using conda:

    ```shell
    conda env create -f environment.yml
    ```

3. Install the development environment with conda:
    ```shell
    conda env create -f dev-environment.yml
    ```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [github discussions][ark-spatial-github-discussions].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/angelolab/ark-spatial/issues
[changelog]: https://ark-spatial.readthedocs.io/latest/changelog.html
[link-docs]: https://ark-spatial.readthedocs.io
[link-api]: https://ark-spatial.readthedocs.io/latest/api.html
