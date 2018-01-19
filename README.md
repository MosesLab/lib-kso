# lib-kso
Analysis library for all KSO instruments

## Installation

### Bazel
[Bazel](https://www.bazel.build/) is the main build tool for the project. It allows for easily updating C++ libraries from inside a Python project.

It can be installed on Mint 18 using the following instructions which were taken from [here](https://docs.bazel.build/versions/master/install-ubuntu.html)
```
sudo apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel
```
