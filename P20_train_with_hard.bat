set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_P_net20.py --lr 0.001 --image_set train_20_with_hard_1 --end_epoch 8 --prefix model/pnet20_hard --lr_epoch 4,8 --batch_size 200 --frequent 1000 --thread_num 20
python example\train_P_net20.py --lr 0.00001 --image_set train_20_with_hard_1 --end_epoch 32 --pretrained model/pnet20_hard --epoch 8 --begin_epoch 8 --resume --prefix model/pnet20_hard --lr_epoch 100,200 --batch_size 50000 --frequent 10 --thread_num 20
pause