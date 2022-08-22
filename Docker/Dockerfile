##### General instruction: ######
#docker build -f ./Dockerfile -t opera-adt/rtc:interface . #To build the docker image



#TODO Try to integrate environment.yml and requirements.txt

FROM ubuntu:22.04

LABEL author="OPERA ADT" \
      description="RTC interface release" \
      version="interface_0.1"

RUN apt-get -y update &&\
    apt-get -y install curl git zip make &&\
    adduser --disabled-password rtc_user

USER rtc_user 

COPY Docker/environment.yml /home/rtc_user/
COPY Docker/requirements.txt /home/rtc_user/

ENV CONDA_PREFIX=/home/rtc_user/miniconda3

WORKDIR /home/rtc_user
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniconda.sh

ENV PATH="${CONDA_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
ENV POSTFIX_ISCE3="rtc2"
ENV ISCEHOME=/home/rtc_user

RUN $CONDA_PREFIX/bin/conda init bash


#Install ISCE3 from source code
# TODO modify the URL of the ISCE3 source code when Gustavo forks out
RUN echo "Installing ISCE3 for RTC workflow" &&\
    mkdir -p ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/src &&\
    mkdir -p ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/build &&\
    mkdir -p ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/install &&\
    cd ${ISCEHOME} &&\
    curl -sSL https://github.com/gshiroma/isce3/archive/refs/heads/opera_rtc.zip -o isce3_compass_rtc_workflow.zip &&\
    unzip -d ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3} isce3_compass_rtc_workflow.zip &&\
    cd ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/src &&\
    ln -s ../isce3-opera_rtc isce3 &&\
    rm ${ISCEHOME}/isce3_compass_rtc_workflow.zip &&\
    cd ${HOME} &&\
    conda env create --file environment.yml &&\
    cd ${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/src

SHELL ["conda", "run", "-n", "isce3_rtc", "/bin/bash", "-c"]

ENV CC=${CONDA_PREFIX}/envs/isce3_rtc/bin/x86_64-conda-linux-gnu-gcc
ENV CXX=${CONDA_PREFIX}/envs/isce3_rtc/bin/x86_64-conda-linux-gnu-g++

RUN cd ${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/build &&\
cmake -DCMAKE_INSTALL_PREFIX=${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/install -DWITH_CUDA=OFF ${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/src/isce3 &&\
make -j8 VERBOSE=ON &&\
make install

RUN echo 'PATH : ' ${CONDA_PREFIX} &&\
cd /home/rtc_user/miniconda3/envs/isce3_rtc/lib &&\
ln -s $ISCEHOME/tools/isce_${POSTFIX_ISCE3}/install/lib64/libisce3.so &&\
echo 'ISCE3 installation successful!'

ENV PATH=$PATH:${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/install/bin
ENV PYTHONPATH=$PYTHONPATH:${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/install/packages
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/install/lib


RUN echo "Installing OPERA s1-reader" &&\
    mkdir -p $HOME/OPERA &&\
    cd $HOME/OPERA &&\
    curl -sSL https://github.com/seongsujeong/s1-reader/archive/refs/heads/correction_and_calibration.zip -o s1_reader.zip &&\
    unzip s1_reader.zip &&\
    ln -s s1-reader-correction_and_calibration s1-reader &&\
    rm s1_reader.zip &&\
    python -m pip install ./s1-reader

#COPY . /home/rtc_user/OPERA/rtc

RUN cd $HOME/OPERA &&\
curl -sSL https://github.com/seongsujeong/RTC/archive/refs/heads/setup.zip -o rtc.zip &&\
unzip rtc.zip &&\
ln -s RTC-setup RTC &&\
pip install ./RTC


RUN mkdir /home/rtc_user/scratch &&\
conda install -c conda-forge backoff


WORKDIR /home/rtc_user/scratch

#for COMPASS 0.1.1 and future releases
ENTRYPOINT ["conda", "run", "-n", "isce3_rtc","rtc_s1.py"]
