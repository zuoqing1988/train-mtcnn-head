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
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    #cur size: 9x9

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), num_filter=8, num_group=8, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")
    #cur size: 7x7
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")
    #cur size: 3x3

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    #cur size: 1x1

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=conv4_1, mode="channel", name="cls_prob")
        bbox_pred = conv4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = conv4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = conv4_2, shape=(-1, 4), name="conv4_2_reshape")
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

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1,1), num_filter=16, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=2, name="conv5_1")
    conv5_2 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=4, name="conv5_2")

    cls_prob = mx.symbol.SoftmaxOutput(data=conv5_1, label=label, use_ignore=True,
                                       name="cls_prob")
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxOutput(data=conv5_1, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = conv5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv5_2, label=bbox_target,
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
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=16, name="conv1")
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep", no_bias=True)
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv5_dw", no_bias=True)
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=128, name="conv5_sep", no_bias=True)
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=128, num_group=128, name="conv6_dw", no_bias=True)
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
    conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
    bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")

    conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
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
