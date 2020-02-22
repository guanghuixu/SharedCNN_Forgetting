```
mkdir checkpoints
CUDA_VISIBLE_DEVICES=0 python train_A.py
CUDA_VISIBLE_DEVICES=0 python train_B.py --share_n 1
```
![output.jpg](https://github.com/guanghuixu/SharedCNN_Forgetting/blob/master/output.png)