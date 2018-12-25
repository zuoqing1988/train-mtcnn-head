set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python prepare_data/gen_hard_example.py --test_mode onet --prefix model/pnet20_hard,model/rnet --epoch 16,16 --thresh 0.6,0.5
pause