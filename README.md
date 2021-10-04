![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
![Build](https://github.com/fractal-napari-plugins-collection/napari_tissuemaps_interface/actions/workflows/build.yml/badge.svg?branch=master)

# Napari - Plugins
This repository contains the following Napari plugin developed for the Fractal
Analytics Platform (https://github.com/fractal-analytics-platform):

| Plugin | Description |
| :--- | :--- |
| napari_tissuemaps_interface | Widget to stream images from a TissueMAPS server|


> The plugin is not bound to the Fractral Analytics Platform and can be used
> standalone.

## Installation 
To avoid conflicts with other packages, it is recommended to install this
package within a virtual environment. Please refer to "The Python Tutorial"
for further information (https://docs.python.org/3/tutorial/venv.html).

The plugin can be installed from source either by calling setup.py directly...

```
python setup.py
```

...or, via pip...

```
pip install .
```

## Usage

The widget is available via "Plugins - Add Dock Widget".

> You can find all installed plugins from Napari's
> "Plugins - Install/Uninstall Package(s)..." menu.


## Tests
This repository comes with a set of automated tests that can be run by the
following command:

```
python setup.py test
```

## Documentation
The user documentation is available via https://fractal-napari-plugins-collection.github.io/user-documentation/.

## Copyright
Copyright (c) 2021, Friedrich Miescher Institute for Biomedical Research & University of Zurich. All rights reserved.
Licensed under BSD 3-Clause - see ./LICENSE
