set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_P_net20.py --gpus 0 --lr 0.001 --image_set train_20_1 --prefix model/pnet20 --end_epoch 16 --lr_epoch 8,14 --frequent 10 --batch_size 1000 --thread_num 24
pause