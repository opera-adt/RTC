FROM oraclelinux:8

LABEL author="OPERA ADT" \
    description="RTC cal/val release R4" \
    version="0.4.2-calval"

RUN yum -y update &&\
    yum -y install curl &&\
    adduser rtc_user

RUN mkdir -p /home/rtc_user/OPERA/RTC

RUN chmod -R 755 /home/rtc_user &&\
    chown -R rtc_user:rtc_user /home/rtc_user/OPERA

USER rtc_user

ENV CONDA_PREFIX=/home/rtc_user/miniconda3

# install Miniconda
WORKDIR /home/rtc_user
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniconda.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN ${CONDA_PREFIX}/bin/conda init bash

# copy RTC source code and set rtc_user as owner
COPY --chown=rtc_user:rtc_user . /home/rtc_user/OPERA/RTC

# create CONDA environment
RUN conda create --name "RTC" --file /home/rtc_user/OPERA/RTC/Docker/lockfile.lock &&  conda clean -afy

SHELL ["conda", "run", "-n", "RTC", "/bin/bash", "-c"]

WORKDIR /home/rtc_user/OPERA

# installing OPERA RTC
RUN python -m pip install ./RTC &&\
    echo "conda activate RTC" >> /home/rtc_user/.bashrc

WORKDIR /home/rtc_user/scratch

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "RTC"]
