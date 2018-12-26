import numpy as np
import mxnet as mx
import threading
import argparse
import os
import cv2
import sys
import math
sys.path.append(os.getcwd())
from config import config
from core.symbol import P_Net20, R_Net, O_Net
from core.imdb import IMDB
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector20 import MtcnnDetector
from utils import *

class MyThread_test(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread_test, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.detections = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.detections
        except Exception:
            return None

class MyThread_gen(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread_gen, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.neg_hard_names = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.neg_hard_names
        except Exception:
            return None

def test_net_thread(imdb, mtcnn_detector):
    test_data = TestLoader(imdb)
    detections = mtcnn_detector.detect_face(imdb, test_data, vis=False)
    return detections

def creat_mtcnn_detector(prefix, epoch, batch_size, test_mode, thresh, min_face_size, ctx):
    detectors = [None, None, None]
	# load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    PNet = FcnDetector(P_Net20("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["onet", "hardrnet", "hardonet"]:
        args, auxs = load_param(prefix[1], epoch[1], convert=True, ctx=ctx)
        RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
        detectors[1] = RNet

    # load onet model
    if test_mode == "hardonet":
        args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
        ONet = Detector(O_Net("test",False), 48, batch_size[2], ctx, args, auxs)
        detectors[2] = ONet
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size, stride=4, threshold=thresh, slide_window=False)
    return mtcnn_detector
	
def test_net(root_path, dataset_path, image_set, prefix, epoch, batch_size, ctx, 
            test_mode="hardpnet20",thresh=[0.6, 0.6, 0.7], min_face_size=24):

    thread_num = len(ctx)
    mtcnn_detectors = list()
    for i in range(thread_num):
        mtcnn_detectors.append(creat_mtcnn_detector(prefix,epoch,batch_size,test_mode,thresh,min_face_size,ctx[i]))
    
    imdb = IMDB("wider", image_set, root_path, dataset_path, 'test')
    annotations = imdb.get_annotations()

    image_num = len(annotations)
    test_batch_size = 10*thread_num
    start_idx = 0
    detections = list()
    while start_idx < image_num:
        end_idx = min(start_idx+test_batch_size,image_num)
        cur_annotations = annotations[start_idx:end_idx]
        cur_detections = test_minibatch(cur_annotations,mtcnn_detectors)
        detections = detections+cur_detections
        start_idx = end_idx
        print '%d images done'%start_idx

    return detections

def test_minibatch(imdb, mtcnn_detectors):
    num_images = len(imdb)
    thread_num = len(mtcnn_detectors)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    #print(num_per_thread)
    if thread_num == 1:
        detections = test_net_thread(imdb,mtcnn_detectors[0])
    else:
        threads = []
        for t in range(thread_num):
            start_idx = int(num_per_thread*t)
            end_idx = int(min(num_per_thread*(t+1),num_images))
            cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
            cur_thread = MyThread_test(test_net_thread,(cur_imdb,mtcnn_detectors[t]))
            threads.append(cur_thread)
        for t in range(thread_num):
            threads[t].start()

        detections = list()
    
        for t in range(thread_num):
            cur_detections = threads[t].get_result()
            detections = detections + cur_detections
    return detections

def save_hard_example(annotation_lines, det_boxes, size, thread_num):

    num_of_images = len(annotation_lines)
    neg_hard_save_dir = "%s/prepare_data/%d/negative_hard"%(config.root,size)
    save_path = "%s/prepare_data/%d"%(config.root,size)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(neg_hard_save_dir):
        os.mkdir(neg_hard_save_dir)
    f = open(os.path.join(save_path, 'neg_hard.txt'), 'w')
    
    #print len(det_boxes)
    #print len(det_boxes[0])
    #print num_of_images
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    minibatch_size = 10*thread_num
    start_idx = 0
    neg_hard_num = 0
    while start_idx < num_of_images:
        end_idx = min(start_idx+minibatch_size,num_of_images)
        cur_annotation_lines = annotation_lines[start_idx:end_idx]
        cur_det_boxes = det_boxes[start_idx:end_idx]
        neg_hard_names = gen_hard_minibatch(size, start_idx, cur_annotation_lines, cur_det_boxes, neg_hard_save_dir, thread_num)
        cur_neg_hard_num = len(neg_hard_names)
        for i in range(cur_neg_hard_num):
            f.write(neg_hard_names[i]+'\n')
        neg_hard_num += cur_neg_hard_num
        start_idx = end_idx
        print '%d images done, neg_hard_num: %d'%(start_idx,neg_hard_num)

    f.close()
	

def gen_hard_minibatch(size, start_idx, annotation_lines, det_boxes, neg_hard_save_dir, thread_num = 4):
    num_images = len(annotation_lines)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    threads = []
    for t in range(thread_num):
        cur_start_idx = int(num_per_thread*t)
        cur_end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_annotation_lines = annotation_lines[cur_start_idx:cur_end_idx]
        cur_det_boxes = det_boxes[cur_start_idx:cur_end_idx]
        cur_thread = MyThread_gen(gen_hard_minibatch_thread,(size, start_idx+cur_start_idx, cur_annotation_lines, cur_det_boxes,
                                                        neg_hard_save_dir))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    neg_hard_names = list()
    
    for t in range(thread_num):
        cur_neg_hard_names = threads[t].get_result()
        neg_hard_names = neg_hard_names + cur_neg_hard_names

    return neg_hard_names
	
def gen_hard_minibatch_thread(size, start_idx, annotation_lines, det_boxes, neg_hard_save_dir):
    num_images = len(annotation_lines)
    neg_hard_names = list()
    for i in range(num_images):
        cur_annotation_line = annotation_lines[i].strip().split(' ')
        im_path = '%s/data/HollywoodHeads/JPEGImages/'%config.root+cur_annotation_line[0]
        bbox = map(float, cur_annotation_line[1:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(im_path)
        cur_neg_hard_names = gen_hard_for_one_image(size, start_idx+i, img, det_boxes[i], boxes, neg_hard_save_dir)
        neg_hard_names = neg_hard_names + cur_neg_hard_names

    return neg_hard_names
	
def gen_hard_for_one_image(size, idx, img, det_boxes, gt_boxes,  neg_hard_save_dir):
    neg_hard_num = 0
    neg_hard_names = list()
    
    det_boxes = np.array(det_boxes,dtype=np.float32)
    if det_boxes.shape[0] == 0:
        return neg_hard_names

    dets = convert_to_square(det_boxes)
    dets[:, 0:4] = np.round(dets[:, 0:4])

    for box in dets:
        #print box
        x_left, y_top, x_right, y_bottom,_= box.astype(int)
        width = x_right - x_left + 1
        height = y_bottom - y_top + 1

        # ignore box that is too small or beyond image border
        if width < 20 or height < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gt_boxes)
        cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

        # save negative images and write label
        if np.max(Iou) < 0.1:
            # Iou with all gt_boxes must below 0.1
            save_file = os.path.join(neg_hard_save_dir, '%d_%d.jpg'%(idx,neg_hard_num))
            if cv2.imwrite(save_file, resized_im):
                line = '%s/%d_%d 0'%(neg_hard_save_dir,idx,neg_hard_num)
                neg_hard_names.append(line)
                neg_hard_num += 1

    return neg_hard_names

def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='%s/data'%config.root, type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='%s/data/mtcnn'%config.root, type=str)
    parser.add_argument('--image_set', dest='image_set', help='image set',
                        default='train', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be hardpnet20, rnet, hardrnet or onet, hardonet',
                        default='hardpnet20', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name',
                        default='%s/model/pnet20'%config.root+',%s/model/rnet'%config.root+',%s/model/onet'%config.root, type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', 
                        default='16,16,16', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', 
                        default='2048,256,16', type=str)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet',
                        default='0.3,0.3,0.3', type=str)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--target_size', dest='target_size', help='target_size',
                        default=-1, type=int)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num',
                        default=4, type=int)
    parser.add_argument('--gpus', dest='gpus', help='GPU device to train with',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    prefix = args.prefix.split(',')
    epoch = [int(i) for i in args.epoch.split(',')]
    batch_size = [int(i) for i in args.batch_size.split(',')]
    thresh = [float(i) for i in args.thresh.split(',')]
    test_mode = args.test_mode
    detections = test_net(args.root_path, args.dataset_path, args.image_set, prefix, epoch, batch_size, ctx, 
            test_mode, thresh, args.min_face)
    
    anno_file = "%s/prepare_data/annotations/anno.txt"%config.root
    with open(anno_file, 'r') as f:
        annotation_lines = f.readlines()
		
    if test_mode == "hardpnet20":
        size = 20
    elif test_mode == "rnet" or test_mode == "hardrnet":
        size = 24
    elif test_mode == "onet" or test_mode == "hardonet":
        size = 48
    if args.target_size > 0:
        size = args.target_size
    save_hard_example(annotation_lines, detections, size, args.thread_num)
