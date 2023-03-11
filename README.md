
# SSAD: source separation followed by anomaly detection

This is an official repository for our paper, "Activity-informed Industrial Audio Anomaly Detection via Source Separation".

If you are considering using repository, please cite our paper:
```
@inproceedings{kim2023activity,
  title={Activity-informed Industrial Audio Anomaly Detection via Source Separation},
  author={Jaechang Kim and Yunjoo Lee and Hyun Mi Cho and Dong Woo Kim and Chi Hoon Song and Jungseul Ok},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2023}
}
```


# Environment Setting
```base
conda env create -n asteroid
conda activate asteroid
conda install python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# for asteroid
pip install -r requirements/dev.txt
pip install -e .
pip install torchmetrics==0.6.0

# for anomaly detection
pip install -r anomaly/requirements.txt

```

# Training Source Separation Models
To run X-UMX change the configuration file considering the type of data and the use of control signal.
## 1. First change the configuration 
```bash
cd informed-X-UMX
vi local/conf_???.yml
```

Edit `local/conf_base.yml` for XUMX baseline and `local/conf_informed.yml` for informed source separation model.

* data:train_dir -> MIMII dataset directory
* data:output -> directory where checkpoint and log files will be saved
* data:machine_type -> machine types to use
* data:sources -> machine ids to use
* model:pretrained -> pretrained model path

## 2. train the model by running
```bash
cd informed-X-UMX
train.py --conf local/conf_base.yml
```
Run train.py for with given configuration file.


# Anomaly Detection models

Edit `anomaly/baseline.yaml`

* base_directory -> MIMII dataset path

## Oracle baseline

Edit `anomaly/baseline.py`

* Check datapath near line 196
  * dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/valve/id_00".format(base=param["base_directory"]))))
  * Choose which machines (types, id) to use

```bash
cd anomaly
python baseline.py
```


## Mixture baseline

Edit `anomaly/baseline_mix.py`

* Check datapath near line 228
  * dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/valve/id_00".format(base=param["base_directory"]))))
  * Choose which machines (types, id) to use
* Check machine_types near line 42
  * Those machine types will be used to make a mixture

```bash
cd anomaly
python baseline_mix.py
```

## SSAD (Proposed Method)


Edit `anomaly/baseline_src_xumx_original.py`

* Check datapath near line 318
  * dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/valve/id_00".format(base=param["base_directory"]))))
  * Choose which machines (types, id) to use
* Check trained separation model path near 363
* Check conf near line 43
  * S1, S2 -> machine id
  * FILE -> AE model path (to save)
* 

```bash
cd anomaly
python baseline_src_xumx_original.py
```

# Acknowledgement

This repository is based on 

* https://github.com/asteroid-team/asteroid
* https://github.com/MIMII-hitachi/mimii_baseline

