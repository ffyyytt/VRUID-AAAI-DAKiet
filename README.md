# VRUID-AAAI-DAKiet
First place solution at [AAAI-25 Visually-Rich Document(VRD-IU) Leaderboard](https://www.kaggle.com/competitions/aaai-25-visually-rich-document-vrd-iu-leaderboard)

If you find the source codes useful and involve them in your research, please consider citing our paper

```BibTeX
@article{duong2025,
    title     = {Hierarchical Document Parsing via Large Margin Feature Matching and Heuristics},
    author    = {Duong, Anh-Kiet},
    journal   = {}, 
    volume    = {},
    number    = {},
    year      = {2025}
}
```

## Training
data:
```console
|-- dataset/
    |-- *.pkl
    |-- train
        |-- train
            |-- *.png
    |-- val
        |-- val
            |-- *.png
    |-- test
        |-- test
            |-- *.png
```

Run:
```console
python train.py -category table
python train.py -category figure
python train.py -category form
python train.py -category list
python train.py -category form_body
```

## Eval
Please refer to
* Model: [https://www.kaggle.com/models/fdfyaytkt/vruid-aaai-dakiet/PyTorch/figure/2](https://www.kaggle.com/models/fdfyaytkt/vruid-aaai-dakiet/PyTorch/figure/2)
* Code: [https://www.kaggle.com/code/fdfyaytkt/dakiet-aaai25-vruid](https://www.kaggle.com/code/fdfyaytkt/dakiet-aaai25-vruid)

## 
