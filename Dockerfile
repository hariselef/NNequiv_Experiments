# syntax=docker/dockerfile:1

FROM continuumio/anaconda3

WORKDIR C:/Users/haris/Desktop/test/

# # Create the environment:
COPY src src/
COPY models models/
COPY environment.yml .

SHELL [ "/bin/bash", "--login", "-c" ]

# Make non-activate conda commands available.
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda activate command available from /bin/bash --login shells.
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# Make conda activate command available from /bin/bash --interative shells.
RUN conda init bash

# Create and activate the environment.
RUN conda env create --force -f environment.yml
RUN echo "conda activate nnequiv" >> ~/.profile
 
