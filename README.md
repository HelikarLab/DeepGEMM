<div align="center">
  <img src=".github/assets/logo.png" height="128">
  <h1>
      gempy
  </h1>
  <h4>Genome-Scale Metabolic Modelling</h4>
</div>

<p align="center">
    <a href='https://github.com/achillesrasquinha/gempy//actions?query=workflow:"Continuous Integration"'>
      <img src="https://img.shields.io/github/workflow/status/achillesrasquinha/gempy/Continuous Integration?style=flat-square">
    </a>
    <a href="https://coveralls.io/github/achillesrasquinha/gempy">
      <img src="https://img.shields.io/coveralls/github/achillesrasquinha/gempy.svg?style=flat-square">
    </a>
    <a href="https://pypi.org/project/gempy/">
      <img src="https://img.shields.io/pypi/v/gempy.svg?style=flat-square">
    </a>
    <a href="https://pypi.org/project/gempy/">
      <img src="https://img.shields.io/pypi/l/gempy.svg?style=flat-square">
    </a>
    <a href="https://pypi.org/project/gempy/">
      <img src="https://img.shields.io/pypi/pyversions/gempy.svg?style=flat-square">
    </a>
    <a href="https://git.io/boilpy">
      <img src="https://img.shields.io/badge/made%20with-boilpy-red.svg?style=flat-square">
    </a>
</p>

### Table of Contents
* [Features](#features)
* [Quick Start](#quick-start)
* [Usage](#usage)
* [License](#license)

### Features
* Python 2.7+ and Python 3.4+ compatible.

### Quick Start

```shell
$ pip install gempy
```

Check out [installation](docs/source/install.rst) for more details.

### Usage

#### Application Interface

```python
>>> import gempy
```


#### Command-Line Interface

```console
$ gempy
Usage: gempy [OPTIONS] COMMAND [ARGS]...

  Genome-Scale Metabolic Modelling

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  help     Show this message and exit.
  version  Show version and exit.
```


### Docker

Using `gempy's` Docker Image can be done as follows:

```
$ docker run \
    --rm \
    -it \
    achillesrasquinha/gempy \
      --verbose
```

### License

This repository has been released under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ using <a href="https://git.io/boilpy">boilpy</a>.
</div>