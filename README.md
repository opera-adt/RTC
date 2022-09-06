# RTC
OPERA Radiometric Terrain-Corrected (RTC) Product

🚨 This toolbox is still in **pre-alpha** stage and undergoing **rapid development**. 🚨

# Installation

1. Install ISCE3
    Users are suggested to install up-to-date version of ISCE3 from the source code available in the github repository below:
    https://github.com/isce-framework/isce3

2. Install s1-reeader

    ```bash
    python -m pip install git+https://github.com/opera-adt/s1-reader.git
    ```
    Installing `ISCE3` and `s1-reader` should install all dependencies required by `RTC`.

3. Install RTC

     From the repository directory, try:
    ```bash
    python -m pip install .
    ```

# Usage
The workflow is implemented in `RTC/app/rtc_s1.py`
```bash
rtc_s1.py <path to rtc_s1 runconfig yaml file>
```
# License
Copyright (c) 2021 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
