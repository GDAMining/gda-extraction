# GDA Extraction
This repository contains the source code to train and test Biomedical Relation Extraction (BioRE) models on GDAb and GDAt datasets. GDAb and GDAt are large-scale, distantly supervised, and manually enhanced datasets for Gene-Disease Association (GDA) extraction. In addition, the repository contains scripts to compute datasets statistics and to convert other BioRE datasets in the required format. GDAb and GDAt datasets are available at: http://doi.org/10.5281/zenodo.5113853.

### Usage

Clone this repository

```bash
git clone https://github.com/NanoGDA/gda-extraction.git
```

Then install all the requirements:

```
pip install -r requirements.txt
```

**Note**: Please choose appropriate PyTorch version based on your machine (related to your CUDA version). For details, refer to https://pytorch.org/. 

Then install the OpenNRE package with 
```
cd ./OpenNRE
python setup.py install 
```

If users also want to modify the code, run this instead:
```
cd ./OpenNRE
python setup.py install
python setup.py develop
```

## Datasets 

Users can go into the `benchmark` folder and download <b>GDAb</b> and <b>GDAt</b> datasets using the script `download_GDA_datasets.sh`. 

If interested in running models on BioRel and DTI, users can download and store these datasets as follows.

<b>BioRel</b>: <br />
Download in `/benchmark/biorel/` the `train.json`, `dev.json`, `test.json`, and `relation2id.json` files from https://bit.ly/biorel_dataset. <br />
Then, run the `convert_biorel2opennre.sh` file in `/convert2opennre`.

<b>DTI</b>: <br />
Download in `/benchmark/dti/` the `train.json`, `valid.json`, and `test.json` files from https://cloud.tsinghua.edu.cn/d/c9651d22d3f94fb7a4f8/. <br />
Then, run the `convert_dti2opennre.sh` file in `/convert2opennre`.

## Dataset Statistics

Users can compute datasets statistics to understand the differences between datasets. For instance, if a user wants to compute statistics for GDAb, they can run

```
python data_stats.py --benchmark_fpath ./benchmark/GDAb
```

## Pretrain
Pretrained embeddings can be downloaded by running scripts in the ``pretrain`` folders. For instance, if a user wants to download BioWordVec embeddings, they can run

```bash
cd ./pretrain
bash download_biowordvec.sh
```

Once downloaded, pretrained embeddings can be tailored to the considered dataset. For instance, if a user wants to experiment with GDAb, they can run

```
python prepare_embeddings.py --embs_fpath ./pretrain/biowordvec/ --benchmark_fpath ./benchmark/GDAb/
```

## Training

Users can train the provided models on the 
You can train your own models on your own data with OpenNRE. In `example` folder we give example training codes for supervised RE models and bag-level RE models. You can either use our provided datasets or your own datasets. For example, you can use the following script to train a PCNN-ATT bag-level model on the NYT10 dataset with manual test set:
```bash
python example/train_bag_cnn.py \
    --metric auc \
    --dataset nyt10m \
    --batch_size 160 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --max_epoch 100 \
    --max_length 128 \
    --seed 42 \
    --encoder pcnn \
    --aggr att
```

Or use the following script to train a BERT model on the Wiki80 dataset:
```bash
python example/train_supervised_bert.py \
    --pretrain_path bert-base-uncased \
    --dataset wiki80
```

We provide many options in the example training code and you can check them out for detailed instructions.

## Cite

If you use or extend our work, please cite the following:

```
@dataset{marchesin-silvello-2021-gda,
  title = "From Nanopublications to Large-Scale Gene-Disease Association Datasets for Biomedical Relation Extraction",
  author = "S. Marchesin and G. Silvello",
  publisher = "Zenodo",
  year = "2021",
  version = "1.0",
  url = "https://doi.org/10.5281/zenodo.5113853",
  doi = "10.5281/zenodo.5113853"
}
```

```
@inproceedings{han-etal-2019-opennre,
  title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
  author = "X. Han and T. Gao and Y. Yao and D. Ye and Z. Liu and M. Sun",
  booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
  year = "2019",
  url = "https://www.aclweb.org/anthology/D19-3029",
  doi = "10.18653/v1/D19-3029",
  pages = "169--174"
}
```

If you use the BERE RE model or the DTI dataset, please cite the following:

```
@article{hong-etal-2020-bere,
  title = "A novel machine learning framework for automated biomedical relation extraction from large-scale literature repositories",
  author = "L. Hong and J. Lin and S. Li and F. Wan and H. Yang and T. Jiang and D. Zhao and J. Zeng",
  journal = "Nature Machine Intelligence",
  year = "2020",
  url = "https://www.nature.com/articles/s42256-020-0189-y",
  doi = "10.1038/s42256-020-0189-y",
  volume = "2",
  pages = "347--355"	
}
```

If you use the BioRel dataset, please cite the following:

```
@article{xing-etal-2020-biorel,
  title     = "BioRel: towards large-scale biomedical relation extraction",
  author    = "R. Xing and J. Luo and T. Song",
  journal   = "BMC Bioinformatics",
  year      = "2020",
  url = "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03889-5",
  doi = "10.1186/s12859-020-03889-5",
  volume    = "21-S",
  number    = "16",
  pages     = "543"
}
```
