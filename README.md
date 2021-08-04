![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
![Build](https://github.com/fractal-napari-plugins-collection/napari_tissuemaps_interface/actions/workflows/build.yml/badge.svg?branch=master)

# Napari - Plugins
This repository contains the following Napari plugins developed for the Fractal
Analytics Platform (https://github.com/fractal-analytics-platform):

| Plugin | Description |
| :--- | :--- |
| napari_tissuemaps_interface | Widget to stream images from a TissueMAPS server|


> The plugins are not bound to the Fractral Analytics Platform and can be used
> standalone.

## Installation 
To avoid conflicts with other packages, it is recommended to install this
package within a virtual environment. Please refer to "The Python Tutorial"
for further information (https://docs.python.org/3/tutorial/venv.html).

The plugins can be installed from source either by calling setup.py directly...

```
python setup.py
```

...or, via pip...

```
pip install .
```

## Usage
The plugins are automatically discovered by Napari and involved when opening
new files.

> You can find all installed plugins from Napari's
> "Plugins - Install/Uninstall Package(s)..." menu.


## Tests
This repository comes with a set of automated tests that can be run by the
following command:

```
python setup.py test
```

## Documentation

TBD
[//]: # "Add paragraph on how to generate the Sphinx documentation."

## Copyright
The copyright holders do NOT offer any license for this project.
This means as nobody but the copyright holders themselves can use, copy, distribute, or modify the project!
Please note the difference between UNLICENSED and UNLICENSE projects (which mean the opposite).

@see also: https://choosealicense.com/no-permission/
