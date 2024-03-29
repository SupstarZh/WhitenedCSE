## WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings [ACL 2023]

This repository contains the code and pre-trained models for our paper [WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2305.17746).


Our code is mainly based on the code of SimCSE. Please refer to their repository for more detailed information.

## Overview
We presents a whitening-based contrastive learning method for sentence embedding learning (WhitenedCSE), which combines contrastive learning with a novel shuffled group whitening.

![](./figure/model.png)



## Train WhitenedCSE

In the following section, we describe how to train a WhitenedCSE model by using our code.

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.12.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.12.1` should also work. 

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```
For unsupervised WhitenedCSE, we sample 1 million sentences from English Wikipedia; You can run `data/download_wiki.sh` to download the two datasets.

download the dataset 
```bash 
./download_wiki.sh
```


### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. 

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```


```bash
CUDA_VISIBLE_DEVICES=[gpu_ids]\
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-whitenedcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 1e-5 \
    --num_pos 3 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --dup_type bpe \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
```


Then come back to the root directory, you can evaluate any `transformers`-based pre-trained models using our evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path <your_output_model_dir>  \
    --pooler cls \
    --task_set sts \
    --mode test
```
which is expected to output the results in a tabular format:
```
# BERT-base-uncased
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 74.03 | 84.90 | 76.40 | 83.40 | 80.23 |    81.14     |      71.33      | 78.78 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+

# BERT-large
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 74.65 | 85.79 | 77.49 | 84.71 | 80.33 |    81.48     |      75.34      | 79.97 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

## Pretrained Model


* WhitenedCSE-BERT-base-uncased: https://huggingface.co/SupstarZh/whitenedcse-bert-base
* WhitenedCSE-BERT-large: https://huggingface.co/SupstarZh/whitenedcse-bert-large



## Citation

Please cite our paper if you use WhitenedCSE in your work:

```bibtex
@inproceedings{zhuo2023whitenedcse,
  title={WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings},
  author={Zhuo, Wenjie and Sun, Yifan and Wang, Xiaohan and Zhu, Linchao and Yang, Yi},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={12135--12148},
  year={2023}
}
```

