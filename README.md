# IRIS Despike
## Introduction
Despiking code intended for use on IRIS spectra. Based on code developed by Charles Kankelborg and Jacob Parker

## Installation Instructions
### Dependencies
This program depends on tensorflow. Installation instructions using pip are from [here](https://www.tensorflow.org/install/install_linux#InstallingNativePip), and summarized below
```
sudo apt-get install python3-pip python3-dev
pip3 install setuptools tensorflow-gpu
```
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
sudo apt install python3.4-dev
```
Not sure how to install the dependencies above for 3.4 yet
