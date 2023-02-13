FROM ubuntu:22.04

LABEL author="OPERA ADT" \
      description="RTC gamma release" \
      version="0.3-gamma"

RUN apt-get -y update &&\
    apt-get -y install curl zip make &&\
    adduser --disabled-password rtc_user

RUN mkdir -p /home/rtc_user/OPERA/RTC
COPY . /home/rtc_user/OPERA/RTC
RUN chmod -R 755 /home/rtc_user &&\
    chown -R rtc_user:rtc_user /home/rtc_user/OPERA
USER rtc_user 

ENV CONDA_PREFIX=/home/rtc_user/miniconda3


# Install conda and RTC
WORKDIR /home/rtc_user
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    chmod -R 755 ${HOME}/miniconda3 &&\
    rm ${HOME}/miniconda.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN ${CONDA_PREFIX}/bin/conda init bash

RUN conda create --name "RTC" --file /home/rtc_user/OPERA/RTC/Docker/specfile.txt
SHELL ["conda", "run", "-n", "RTC", "/bin/bash", "-c"]


RUN echo "Installing OPERA s1-reader and RTC" &&\
    cd $HOME/OPERA &&\
    curl -sSL https://github.com/opera-adt/s1-reader/archive/refs/tags/v0.1.3.zip -o s1_reader.zip &&\
    unzip s1_reader.zip &&\
    chmod -R 755 s1-reader-0.1.3 &&\
    ln -s s1-reader-0.1.3 s1-reader &&\
    rm s1_reader.zip &&\
    python -m pip install ./s1-reader &&\
    python -m pip install ./RTC &&\
    mkdir /home/rtc_user/scratch &&\
    chmod -R 755 /home/rtc_user/scratch &&\
    echo 'conda activate RTC' >>/home/rtc_user/.bashrc

WORKDIR /home/rtc_user/scratch

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "RTC","rtc_s1.py"]
