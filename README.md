# d-VAE

This is official implementation of paper: *Revealing unexpected complex encoding but simple decoding mechanisms in motor cortex via separating behaviorally relevant neural signals*

## Requirements

To install requirements:

```shell script
pip install -r requirements.txt
```

## Training

```shell script
python main_dvae.py && python main_dvae_restore.py && python main_ann.py && python main_ann_restore.py && python main_kf.py
```

## Bib

```latex
@article{li2024revealing,
  title={Revealing unexpected complex encoding but simple decoding mechanisms in motor cortex via separating behaviorally relevant neural signals},
  author={Li, Yangang and Zhu, Xinyun and Qi, Yu and Wang, Yueming},
  journal={eLife},
  volume={12},
  year={2024},
  publisher={eLife Sciences Publications Limited}
}
```
