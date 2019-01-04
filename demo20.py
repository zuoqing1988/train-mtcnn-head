import numpy as np
import mxnet as mx
import argparse
import cv2
import time
from core.symbol import P_Net20, R_Net, O_Net
from core.fcn_detector import FcnDetector
from core.detector import Detector
from tools.load_model import load_param
from core.MtcnnDetector20 import MtcnnDetector


def test_net(imgfile,prefix, epoch, batch_size, ctx,
             thresh=[0.6, 0.6, 0.7], min_face_size=24):

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    PNet = FcnDetector(P_Net20("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[1], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=4, threshold=thresh, slide_window=False)

    img = cv2.imread(imgfile)
    t1 = time.time()

    boxes, boxes_c = mtcnn_detector.detect_pnet20(img)
    boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
    boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)

    print 'time: ',time.time() - t1

    if boxes_c is not None:
        draw = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for b in boxes_c:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            cv2.putText(draw, '%.3f'%b[4], (int(b[0]), int(b[1])), font, 0.4, (0, 0, 255), 1)
        cv2.imwrite('demoresult.jpg',draw)
        cv2.imshow("detection result", draw)
        cv2.waitKey(0)



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imgfile', dest='imgfile', help='image filename', 
                        default='4.jpg', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default='model/pnet20_hard,model/rnet,model/onet', type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load',
                        default='16,16,16', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', 
                        default='2048,256,16', type=str)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', 
                        default='0.5,0.5,0.7', type=str)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=20, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = mx.gpu(args.gpu_id)
    if args.gpu_id == -1:
        ctx = mx.cpu(0)
    prefix = args.prefix.split(',')
    epoch = [int(i) for i in args.epoch.split(',')]
    batch_size = [int(i) for i in args.batch_size.split(',')]
    thresh = [float(i) for i in args.thresh.split(',')]
    test_net(args.imgfile,prefix, epoch, batch_size, ctx, thresh, args.min_face)
