# NNequiv_Experiments

This repository contains the experimental results for our paper "On Neural Network Equivalence Checking using SMT solvers"


### How to run #

1) You have to download and enable docker in your local machine. You can find more info at https://docs.docker.com/get-docker/

2) Clone the repository to your local machine.
E.g. with Git you can 

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
