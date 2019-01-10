import mxnet as mx
import negativemining
from config import config


def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1") #20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    
    conv2_dw = mx.symbol.Convolution(data=pool1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw") #9/4
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(2, 2), num_filter=64, num_group=64, name="conv3_dw")#4/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=128, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=128, num_group=128, name="conv4_dw") #3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")
    
    conv4_1 = mx.symbol.Convolution(data=prelu4, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1") #24/22
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #22/11

    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=64, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2") #11/5

    conv3_sep = mx.symbol.Convolution(data=pool2, kernel=(1, 1), num_filter=64, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=64, num_group=64, name="conv4_dw") #5/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw") #3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=128, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")

    conv5_1 = mx.symbol.FullyConnected(data=prelu5, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    conv5_2 = mx.symbol.FullyConnected(data=prelu5, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)

    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True,
                                       name="cls_prob")
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group

def O_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1")#48/46
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #46/23

    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=64, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2") #23/11

    conv3_dw = mx.symbol.Convolution(data=pool2, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw") #11/5
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=64, num_group=64, name="conv4_dw") #5/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")
	
    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")

    conv6_1 = mx.symbol.FullyConnected(data=prelu5, num_hidden=2, name="conv6_1")
    bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")

    conv6_2 = mx.symbol.FullyConnected(data=prelu5, num_hidden=4, name="conv6_2")	
    bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
    if mode == "test":
        bbox_pred = bn6_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                 grad_scale=1, name="bbox_pred")
        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                           op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group
