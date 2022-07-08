# NNequiv_Experiments

This repository contains the experimental results for our paper "On Neural Network Equivalence Checking using SMT solvers"


## Software Installation #

### The steps needed to install the software are listed below.

1) Install docker from https://docs.docker.com/get-docker/. Please choose the docker version based on the OS and specifications of your machine.

2) Make sure that docker is working correctly and is enabled/running. In macOS and Windows, this can be done simply by double clicking on the Docker app. In Linux, the user might need to run from the terminal e.g.

```
service docker start
```

3) Download or Clone current repository. In case you have Git installed you can run from terminal:
```
git clone https://github.com/hariselef/NNequiv_Experiments.git 
```
or
```
git clone git@github.com:hariselef/NNequiv_Experiments.git
```

4) Once you have downloaded the source code (from the repository), you need to open a terminal (console, command window, command prompt) and move to the directory you have placed the code, e.g.
```
cd NNequiv Experiments
```
5) he next step is to create (build) the docker container. This step depends heavily on the computer resources and the internet speed. It might vary from 10 minutes up to 1 hour. Run in the terminal:

```
docker image build -t nnequiv .
```

## Run the code
### In the previous section, we downloaded and setup the docker container. The commands to run the code as displayed below. #

1) Run the docker via
```
docker run -it nnequiv
```

2) Setup the anaconda environment that contains the necessary python libraries/dependencies with

```
conda activate nnequiv
```

3) Run individual scripts which contain the results reported in the paper. Each script corresponds to a Table. You can type 

```
python src/<Table name>.py

e.g. python src/Table_1.py
```
Available Table names:
Table_1, Table_2, Table_3, Table_4, Table_5, Table_6, Table_7 & Table_9
