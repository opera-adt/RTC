FROM ubuntu:22.04

LABEL author="OPERA ADT" \
      description="RTC interface release" \
      version="interface_0.1"

RUN apt-get -y update &&\
    apt-get -y install curl zip make &&\
    adduser --disabled-password rtc_user

RUN mkdir -p /home/rtc_user/OPERA/RTC
COPY . /home/rtc_user/OPERA/RTC
RUN chmod -R 755 /home/rtc_user &&\
    chown -R rtc_user:rtc_user /home/rtc_user/OPERA
USER rtc_user 

ENV CONDA_PREFIX=/home/rtc_user/miniconda3


#Install conda and 
WORKDIR /home/rtc_user
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    chmod -R 755 ${HOME}/miniconda3 &&\
    rm ${HOME}/miniconda.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}
ENV POSTFIX_ISCE3="rtc"
ENV ISCEHOME=/home/rtc_user

RUN ${CONDA_PREFIX}/bin/conda init bash

#Install ISCE3 from source code
RUN echo "Installing Conda environment for ISCE3" &&\
    mkdir -p ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/src &&\
    mkdir -p ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/build &&\
    mkdir -p ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/install &&\
    cd ${ISCEHOME} &&\
    curl -sSL https://github.com/gshiroma/isce3/archive/refs/heads/opera_rtc.zip -o isce3_opera_rtc.zip &&\
    unzip -d ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3} isce3_opera_rtc.zip &&\
    cd ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/src &&\
    ln -s ../isce3-opera_rtc isce3 &&\
    rm ${ISCEHOME}/isce3_opera_rtc.zip &&\
    cd ${HOME} &&\
    conda create --name isce3_rtc --file ./OPERA/RTC/Docker/specfile.txt &&\
    chmod -R 755 ${CONDA_PREFIX}/envs/isce3_rtc &&\
    cd ${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/src

SHELL ["conda", "run", "-n", "isce3_rtc", "/bin/bash", "-c"]

ENV CC=${CONDA_PREFIX}/envs/isce3_rtc/bin/x86_64-conda-linux-gnu-gcc
ENV CXX=${CONDA_PREFIX}/envs/isce3_rtc/bin/x86_64-conda-linux-gnu-g++
ENV PATH=$PATH:${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/install/bin
ENV PYTHONPATH=$PYTHONPATH:${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/install/packages
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/install/lib

RUN cd ${ISCEHOME}/tools/isce_$POSTFIX_ISCE3/build &&\
    cmake -DCMAKE_INSTALL_PREFIX=${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/install -DWITH_CUDA=OFF ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/src/isce3 &&\
    make -j32 VERBOSE=ON &&\
    make install &&\
    rm -r ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/build &&\
    rm -r ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/isce3-opera_rtc &&\
    chmod -R 755 ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/install &&\
    cd /home/rtc_user/miniconda3/envs/isce3_rtc/lib &&\
    ln -s ${ISCEHOME}/tools/isce_${POSTFIX_ISCE3}/install/lib/libisce3.so &&\
    echo 'ISCE3 installation successful!'

RUN echo "Installing OPERA s1-reader and RTC" &&\
    mkdir -p $HOME/OPERA &&\
    cd $HOME/OPERA &&\
    curl -sSL https://github.com/seongsujeong/s1-reader/archive/refs/heads/correction_and_calibration.zip -o s1_reader.zip &&\
    unzip s1_reader.zip &&\
    chmod -R 755 s1-reader-correction_and_calibration &&\
    ln -s s1-reader-correction_and_calibration s1-reader &&\
    rm s1_reader.zip &&\
    python -m pip install ./s1-reader &&\
    cd $HOME/OPERA &&\
    pip install ./RTC &&\
    mkdir /home/rtc_user/scratch &&\
    echo 'conda activate isce3_rtc' >>/home/rtc_user/.bashrc

WORKDIR /home/rtc_user/scratch

ENTRYPOINT ["conda", "run", "-n", "isce3_rtc","rtc_s1.py"]
