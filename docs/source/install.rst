.. _install:

### Installation

#### Installation via pip

The recommended way to install **gempy** is via `pip`.

```shell
$ pip install gempy
```

For instructions on installing python and pip see “The Hitchhiker’s Guide to Python” 
[Installation Guides](https://docs.python-guide.org/starting/installation/).

#### Building from source

`gempy` is actively developed on [https://github.com](https://github.com//gempy)
and is always avaliable.

You can clone the base repository with git as follows:

```shell
$ git clone https://github.com//gempy
```

Optionally, you could download the tarball or zipball as follows:

##### For Linux Users

```shell
$ curl -OL https://github.com//tarball/gempy
```

##### For Windows Users

```shell
$ curl -OL https://github.com//zipball/gempy
```

Install necessary dependencies

```shell
$ cd gempy
$ pip install -r requirements.txt
```

Then, go ahead and install gempy in your site-packages as follows:

```shell
$ python setup.py install
```

Check to see if you’ve installed gempy correctly.

```shell
$ gempy --help
```