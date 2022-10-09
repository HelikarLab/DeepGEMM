.. _install:

### Installation

#### Installation via pip

The recommended way to install **dgemm** is via `pip`.

```shell
$ pip install dgemm
```

For instructions on installing python and pip see “The Hitchhiker’s Guide to Python” 
[Installation Guides](https://docs.python-guide.org/starting/installation/).

#### Building from source

`dgemm` is actively developed on [https://github.com](https://github.com/achillesrasquinha/dgemm)
and is always avaliable.

You can clone the base repository with git as follows:

```shell
$ git clone https://github.com/achillesrasquinha/dgemm
```

Optionally, you could download the tarball or zipball as follows:

##### For Linux Users

```shell
$ curl -OL https://github.com/achillesrasquinha/tarball/dgemm
```

##### For Windows Users

```shell
$ curl -OL https://github.com/achillesrasquinha/zipball/dgemm
```

Install necessary dependencies

```shell
$ cd dgemm
$ pip install -r requirements.txt
```

Then, go ahead and install dgemm in your site-packages as follows:

```shell
$ python setup.py install
```

Check to see if you've installed dgemm correctly.

```shell
$ dgemm --help
```