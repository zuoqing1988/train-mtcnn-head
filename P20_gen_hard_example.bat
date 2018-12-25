set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python prepare_data/gen_hard_example.py --test_mode hardpnet20 --epoch 16 --thresh 0.5 --thread_num 10
pause