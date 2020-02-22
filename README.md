```
mkdir checkpoints
CUDA_VISIBLE_DEVICES=0 python train_A.py
CUDA_VISIBLE_DEVICES=0 python train_B.py --share_n 1
```
![output.png](https://github.com/guanghuixuSharedCNN_Forgetting/blob/master/output.png)