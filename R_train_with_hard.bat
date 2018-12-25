set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_R_net.py --lr 0.003 --image_set train_24_with_hard_1 --end_epoch 3 --prefix model/rnet --lr_epoch 8,14,100 --batch_size 500 --thread_num 24
python example\train_R_net.py --lr 0.001 --image_set train_24_with_hard_1 --end_epoch 16 --prefix model/rnet --epoch 3 --begin_epoch 3 --resume --lr_epoch 8,14,100 --batch_size 10000 --thread_num 24 --frequent 20
pause 