import numpy as np
import cv2
import threading
import argparse
import math
import os,sys
import numpy.random as npr
from utils import IoU
sys.path.append(os.getcwd())
from config import config

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.pos_names, self.neg_names, self.part_names = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.pos_names, self.neg_names, self.part_names
        except Exception:
            return None

def gen_data_minibatch_thread(size, start_idx, annotation_lines, prob_lines, pos_save_dir, neg_save_dir, part_save_dir, base_num, prob_thresh):
    num_images = len(annotation_lines)
    pos_names = list()
    neg_names = list()
    part_names = list()
    for i in range(num_images):
        cur_annotation_line = annotation_lines[i].strip().split(' ')
        cur_prob_line = prob_lines[i].strip().split(' ')
        im_path = '%s/data/HollywoodHeads/JPEGImages/'%config.root+cur_annotation_line[0]
        bbox = map(float, cur_annotation_line[1:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        prob = map(float, cur_prob_line[1:])
        probs = np.array(prob, dtype=np.float32).reshape(-1, 1)
        img = cv2.imread(im_path)
        cur_pos_names, cur_neg_names, cur_part_names = gen_data_for_one_image(size, start_idx+i, img, pos_save_dir,neg_save_dir,part_save_dir,boxes, probs, base_num, prob_thresh)
        pos_names = pos_names + cur_pos_names
        neg_names = neg_names + cur_neg_names
        part_names = part_names + cur_part_names


    return pos_names, neg_names, part_names


def gen_data_minibatch(size, start_idx, annotation_lines, prob_lines, pos_save_dir, neg_save_dir, part_save_dir, base_num, prob_thresh, thread_num = 4):
    num_images = len(annotation_lines)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    threads = []
    for t in range(thread_num):
        cur_start_idx = int(num_per_thread*t)
        cur_end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_annotation_lines = annotation_lines[cur_start_idx:cur_end_idx]
        cur_prob_lines = prob_lines[cur_start_idx:cur_end_idx]
        cur_thread = MyThread(gen_data_minibatch_thread,(size, start_idx+cur_start_idx, cur_annotation_lines, cur_prob_lines, 
                                                        pos_save_dir, neg_save_dir, part_save_dir, base_num, prob_thresh))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    pos_names = list()
    neg_names = list()
    part_names = list()

    for t in range(thread_num):
        cur_pos_names, cur_neg_names, cur_part_names = threads[t].get_result()
        pos_names = pos_names + cur_pos_names
        neg_names = neg_names + cur_neg_names
        part_names = part_names + cur_part_names

    return pos_names, neg_names, part_names
	

def gen_data_for_one_image(size, idx, img, pos_save_dir,neg_save_dir,part_save_dir,boxes, probs, base_num = 1, prob_thresh = 0.3):
    pos_names = list()
    neg_names = list()
    part_names = list()
    pos_num = 0
    neg_num = 0
    part_num = 0
    
    width = img.shape[1]
    height = img.shape[0]
    while neg_num < base_num*3:
        cur_size = npr.randint(size, min(width, height) / 2)
        nx = npr.randint(0, width - cur_size)
        ny = npr.randint(0, height - cur_size)
        crop_box = np.array([nx, ny, nx + cur_size, ny + cur_size])
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + cur_size, nx : nx + cur_size, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = '%s/%d_%d.jpg'%(neg_save_dir,idx,neg_num)
            if cv2.imwrite(save_file, resized_im):
                line = '%s/%d_%d 0'%(neg_save_dir,idx,neg_num)
                neg_names.append(line)
                neg_num += 1

    box_num = boxes.shape[0]
    for bb in range(box_num):
        box = boxes[bb]
        prob = probs[bb]
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 20 or x1 < 0 or y1 < 0 or prob < prob_thresh:
            continue

        # generate negative examples that have overlap with gt
        for i in range(base_num*1):
            cur_size = npr.randint(size,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-cur_size, -x1), w)
            delta_y = npr.randint(max(-cur_size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + cur_size > width or ny1 + cur_size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + cur_size, ny1 + cur_size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1 : ny1 + cur_size, nx1 : nx1 + cur_size, :]
            resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = '%s/%d_%d.jpg'%(neg_save_dir,idx,neg_num)
                if cv2.imwrite(save_file, resized_im):
                    line = '%s/%d_%d 0'%(neg_save_dir,idx,neg_num)
                    neg_names.append(line)
                    neg_num += 1

        # generate positive examples and part faces
        for i in range(base_num*3):
            cur_size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(x1 + w / 2 + delta_x - cur_size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - cur_size / 2, 0))
            nx2 = nx1 + cur_size
            ny2 = ny1 + cur_size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(cur_size)
            offset_y1 = (y1 - ny1) / float(cur_size)
            offset_x2 = (x2 - nx2) / float(cur_size)
            offset_y2 = (y2 - ny2) / float(cur_size)

            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = '%s/%d_%d.jpg'%(pos_save_dir,idx,pos_num)
                if cv2.imwrite(save_file, resized_im):
                    line = '%s/%d_%d 1 %.2f %.2f %.2f %.2f'%(pos_save_dir,idx,pos_num,offset_x1, offset_y1, offset_x2, offset_y2)
                    pos_names.append(line)
                    pos_num += 1
                
            elif IoU(crop_box, box_) >= 0.4:
                if npr.randint(100) <= 50:
                    save_file = '%s/%d_%d.jpg'%(part_save_dir,idx,part_num)
                    if cv2.imwrite(save_file, resized_im):
                        line = '%s/%d_%d -1 %.2f %.2f %.2f %.2f'%(part_save_dir,idx,part_num,offset_x1, offset_y1, offset_x2, offset_y2)
                        part_names.append(line)
                        part_num += 1

    return pos_names,neg_names,part_names

def gen_data(size=20, base_num = 1, prob_thresh = 0.3, thread_num = 4):
    anno_file = "%s/prepare_data/annotations/anno.txt"%config.root
    prob_file = "%s/prepare_data/annotations/prob.txt"%config.root
    neg_save_dir = "%s/prepare_data/%d/negative"%(config.root,size)
    pos_save_dir = "%s/prepare_data/%d/positive"%(config.root,size)
    part_save_dir = "%s/prepare_data/%d/part"%(config.root,size)

    save_dir = "%s/prepare_data/%d"%(config.root,size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    f1 = open(os.path.join(save_dir, 'pos.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotation_lines = f.readlines()
    with open(prob_file, 'r') as f:
        prob_lines = f.readlines()

    num = len(annotation_lines)
    print "%d pics in total" % num
    batch_size = thread_num*10
    pos_num = 0
    neg_num = 0
    part_num = 0
    start_idx = 0
    while start_idx < num:
        end_idx = min(start_idx+batch_size,num)
        cur_annotation_lines = annotation_lines[start_idx:end_idx]
        cur_prob_lines = prob_lines[start_idx:end_idx]
        pos_names, neg_names, part_names = gen_data_minibatch(size, start_idx, cur_annotation_lines, cur_prob_lines, 
                                                    pos_save_dir, neg_save_dir, part_save_dir, base_num, prob_thresh, thread_num)
        cur_pos_num = len(pos_names)
        cur_neg_num = len(neg_names)
        cur_part_num = len(part_names)
        for i in range(cur_pos_num):
            f1.write(pos_names[i]+'\n')
        for i in range(cur_neg_num):
            f2.write(neg_names[i]+'\n')
        for i in range(cur_part_num):
            f3.write(part_names[i]+'\n')
        pos_num += cur_pos_num
        neg_num += cur_neg_num
        part_num += cur_part_num
        start_idx = end_idx
        print '%s images done, pos: %d neg: %d part: %d'%(end_idx,pos_num,neg_num,part_num)

    f1.close()
    f2.close()
    f3.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', dest='size', help='20 or 24 or 48', default='20', type=str)
    parser.add_argument('--base_num', dest='base_num', help='base num', default='1', type=str)
    parser.add_argument('--prob_thresh', dest='prob_thresh', help='prob thresh', default='0.3', type=str)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num', default='4', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    gen_data(int(args.size), int(args.base_num), float(args.prob_thresh), int(args.thread_num))