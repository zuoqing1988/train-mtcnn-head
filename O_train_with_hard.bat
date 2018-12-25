set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_O_net.py --lr 0.003 --image_set train_48_with_hard_1 --end_epoch 3 --prefix model/onet --lr_epoch 8,14,100 --batch_size 500 --thread_num 24
python example\train_O_net.py --lr 0.001 --image_set train_48_with_hard_1 --end_epoch 16 --prefix model/onet --epoch 3 --begin_epoch 3 --resume --lr_epoch 8,14,100 --batch_size 5000 --thread_num 24 --frequent 20
pause 