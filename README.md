## Download dataset
Download data.tar.gz from [here](https://drive.google.com/file/d/1euERDA5E8CpeCy7RHi2YGQeGRNQwoYeV/view?usp=sharing) and 
```
tar zxvf data.tar.gz
```


## Further pre-training
```bash
bash run_pretrain.sh "GPU number" "dataset"
```

## Fine-tuning from further pre-trained model
```bash 
bash run_finetune.sh "GPU number" "dataset"
```


## Self distillation
After run pre-training the following command.

```bash
bash run_selftrain.sh "GPU number" "dataset" 
```


## Fine-tuning from self-distilled model
```bash 
bash run_finetune.sh "GPU number" "dataset" False self-further-pretrain-20000
```


## Fine-tuning from pre-trained ViT without any further pre-training or self-distillation.
```bash 
bash run_finetune.sh "GPU number" "dataset" True
```
