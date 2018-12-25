set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python prepare_data/gen_hard_example.py --test_mode rnet --prefix model/pnet20_hard --epoch 16 --thresh 0.5
pause