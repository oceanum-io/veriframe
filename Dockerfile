FROM registry.gitlab.com/oceanum/docker/core-ubuntu
MAINTAINER Tom Durrant <t.durrant@oceanum.science>

RUN echo "--------------- Installing packages ---------------" &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get -y install python3-cartopy &&\
    apt-get clean all

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

 # Set required environment variables
ENV REPOS="/source"

COPY setup.py README.rst HISTORY.rst $REPOS/onverify/
COPY onverify $REPOS/onverify/onverify
COPY tests $REPOS/onverify/tests
RUN cd $REPOS/onverify &&\
	pip3 install -e . --no-cache-dir
