# RTC
NASA's Observational Products for End-Users from Remote Sensing Analysis (OPERA) Radiometric Terrain-Corrected (RTC) SAR backscatter from Sentinel-1 (RTC-S1) Science Application Software developed by the OPERA Algoritm Development Team at NASA's Jet Propulsion Laboratory (JPL).



### Install

Instructions to install RTC under a conda environment.

1. Download the source code:

```bash
git clone https://github.com/opera-adt/RTC.git RTC
```

2. Install `isce3`:

```bash
conda install -c conda-forge isce3
```

3. Install `s1-reader` via pip:
```bash
git clone https://github.com/opera-adt/s1-reader.git s1-reader
conda install -c conda-forge --file s1-reader/requirements.txt
python -m pip install ./s1-reader
```

4. Install `RTC` via pip:
```bash
git clone https://github.com/opera-adt/RTC.git RTC
python -m pip install ./RTC
```



### Usage

The command below generates the RTC product:

```bash
rtc_s1.py <path to rtc yaml file>
```

To compare the RTC-S1 products, use `rtc_compare.py`.

```bash
python rtc_s1.py <1st product HDF5> <2nd product HDF5>
```

# License
Copyright (c) 2021 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
