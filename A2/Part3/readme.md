### Installation Guide
#### Conda enviroments
##### Linux
```sh
conda env create -f environment_linux.yml
conda activate cs885a2p3
```
Prepare Dataset
```sh
bash data.sh
```
Run code
```sh
python3 run.py --cql_alpha=1.0
```
##### Mac OS
```sh
conda env create -f environment_mac.yml
conda activate cs885a2p3
```
Prepare Dataset
```sh
bash data.sh
```
Run code
```sh
python3 run.py --cql_alpha=1.0
```

##### Colab
First install condacolab
```python3
!pip install -q condacolab
import condacolab
condacolab.install()
```
Second step import condacolab and check
```python3
import condacolab
condacolab.check()
```
Then, upload cs885a2p3_sk.zip to Colab
```python3
from google.colab import files
files.upload()
```
Unzip this folder
```python3
!unzip cs885a2p3_sk.zip
```
Change current directory to cs885a2p3_sk
```python3
%cd cs885a2p3_sk
```
Update conda base
```python3
!conda env update -n base -f environment_colab.yml
```
Prepare Dataset
```python3
!bash data.sh
```
Run code
```python3
!python3 run.py --cql_alpha=1.0
```
