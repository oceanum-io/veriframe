FROM registry.gitlab.com/oceanum/docker/core-ubuntu:v0.1.3
LABEL maintainer "Tom Durrant <t.durrant@oceanum.science>"

RUN echo "--------------- Installing packages ---------------" &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get -y install libgeos-dev libproj-dev proj-bin proj-data &&\
    apt-get clean all

 # Set required environment variables
ENV REPOS="/source"
COPY auth/oceanum-dev.json $REPOS/auth.json 

COPY setup.py requirements.txt README.rst HISTORY.rst $REPOS/onverify/
COPY onverify $REPOS/onverify/onverify
COPY tests $REPOS/onverify/tests
RUN cd $REPOS/onverify &&\
    pip install -r requirements.txt --no-cache-dir &&\
    pip install -e . --no-cache-dir &&\
    pip install -U aiohttp --no-cache-dir
