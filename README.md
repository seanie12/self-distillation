# Self-Distillation for Further Pre-training of Transformers
This is the official implementation of the paper [**Self-Distillation for Further Pre-training of Transformers**](https://openreview.net/forum?id=kj6oK_Hj40&referrer=%5Bthe%20profile%20of%20Seanie%20Lee%5D(%2Fprofile%3Fid%3D~Seanie_Lee1)).

## Step 1. Download dataset
Download data.tar.gz from [here](https://drive.google.com/file/d/1euERDA5E8CpeCy7RHi2YGQeGRNQwoYeV/view?usp=sharing) and 
```
tar zxvf data.tar.gz
```


## Step 2. Further pre-training. 
Run the following command. The dataset can be either aircraft, chest_xray, cub, dtd, stanford_dogs or  vgg_flower_102.
```bash
bash run_pretrain.sh "GPU number" "dataset"
```

## Step 3. Self distillation
After  further pre-training, run the following command for self-distillation. The dataset can be either aircraft, chest_xray, cub, dtd, stanford_dogs or  vgg_flower_102.

```bash
bash run_selftrain.sh "GPU number" "dataset" 
```

## Step 4.Fine-tuning from self-distilled model
The dataset can be either aircraft, chest_xray, cub, dtd, stanford_dogs or  vgg_flower_102.
```bash 
bash run_finetune.sh "GPU number" "dataset"
```

## (Optional) Fine-tuning from further pre-trained model
The dataset can be either aircraft, chest_xray, cub, dtd, stanford_dogs or  vgg_flower_102.
```bash 
bash run_finetune.sh "GPU number" "dataset" false further-pretrain-20000
```

## (Optional) Fine-tuning from pre-trained ViT without any further pre-training or self-distillation.
The dataset can be either aircraft, chest_xray, cub, dtd, stanford_dogs or  vgg_flower_102.
```bash 
bash run_finetune.sh "GPU number" "dataset" True
```

## Reference
To cite the code/data/paper, please use this BibTex

```bibtex
@inproceedings{
lee2023selfdistillation,
title={Self-Distillation for Further Pre-training of Transformers},
author={Seanie Lee and Minki Kang and Juho Lee and Sung Ju Hwang and Kenji Kawaguchi},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=kj6oK_Hj40}
}
```
