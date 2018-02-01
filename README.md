
# lib-kso
Analysis library for all KSO instruments


# IRIS Despike
## Introduction
Despiking code intended for use on IRIS spectra. Based on code developed by Charles Kankelborg and Jacob Parker

## Installation Instructions

### IDL-Python Bridge
Open the `.bashrc` file 
```
xed ~/.bashrc
```
and add the following to the last line
```
export PYTHONPATH=$PYTHONPATH:/usr/local/exelis/idl/bin/bin.linux.x86_64:/usr/local/exelis/idl/lib/bridges
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/exelis/idl/bin/bin.linux.x86_64

```
IDL 8.5 does not support Python 3.5, we can circumvent this issue by installing Python 3.4

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.4-dev python3.4-tk
```
If you are using PyCharm, make sure to launch it from the command line. This will make sure the environment variables defined in `.bashrc` are inherited by PyCharm.


### Dependencies
This program depends on tensorflow. Installation instructions using pip are from [here](https://www.tensorflow.org/install/install_linux#InstallingNativePip), and summarized below
```
sudo apt-get install python3-pip
python3.4 -m pip install setuptools
python3.4 -m pip install tensorflow-gpu matplotlib
```

```
sudo apt install python3-dev
```
### Boost C++ Libraries
The KSO library depends on a Boost Libraries >1.63 which is not yet available in the `apt` repositories. For now, [download](http://www.boost.org/users/history/version_1_66_0.html), extract and install.
```
cd boost_1_66_0/
sudo ./bootstrap.sh --with-python-version=3.5 --libdir=/usr/lib/x86_64-linux-gnu/ --includedir=/usr/include
sudo ./b2 install -j 8
```

### libCCfits

This library is used for reading FITS files. It should be able to be installed and working with the usual
```
sudo apt install libccfits-dev 
```
However, it appears that this method does not work in Linux Mint 18.0, (while it does in 18.1). 
It can be installed from source by [downloading](https://heasarc.gsfc.nasa.gov/fitsio/CCfits/CCfits-2.5.tar.gz), extracting and then installed for Linux Mint and using the following commands
```
cd CCfits
./configure --includedir=/usr/include --libdir=/usr/lib/x86_64-linux-gnu/
make
make install
```

