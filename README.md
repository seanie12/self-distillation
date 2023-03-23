## Download dataset
Download data.tar.gz and 
```
tar zxvf data.tar.gz
```

## Fine-tuning from pre-trained ViT
```bash 
bash run_finetune.sh "GPU number" "dataset" True
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
