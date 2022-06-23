# NNequiv_Experiments

This repository contains the experimental results for our paper "On Neural Network Equivalence Checking using SMT solvers"


## Software Installation #

### The steps needed to install the software are listed below.

1) Install docker from https://docs.docker.com/get-docker/. Please choose the docker version based on the OS and specifications of your machine.

2) Make sure that docker is working correctly and is enabled/running. In macOS and Windows, this can be done simply by double clicking on the Docker app. In Linux, the user might need to run from the terminal e.g.

```
service docker start
```

```
git clone https://github.com/hariselef/NNequiv_Experiments.git 
```
3) change you current working directory to the project directory.
4) Enable docker.
5) Then type the following command:

```
docker image build -t nnequiv .
```
6) Wait till the build is completed. Then type:

```
docker run -it nnequiv
```

7) Now, you have run dokcer, you have to activate the proper anaconda environment with all the installed dependencies.

```
conda activate nnequiv
```

8) You can generate the results of any Table of the paper by typing:

```
python src/<Table name>.py
```
Available Table names:
Table_1, Table_2, Table_3, Table_4, Table_5, Table_6, Table_7 & Table_9
