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
