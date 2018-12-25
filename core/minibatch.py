import cv2
import threading
from tools import image_processing
import numpy as np
import math

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.labels, self.types, self.bboxes = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.labels, self.types, self.bboxes
        except Exception:
            return None

def get_minibatch_thread(imdb, num_classes, im_size, with_type, with_cls, with_bbox):
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    type_label = list()
    bbox_reg_target = list()
    #print(num_images)
    for i in range(num_images):
        filename = imdb[i]['image']
        #print(filename)
        im = cv2.imread(filename)
        h, w, c = im.shape
        if with_type:
            type = imdb[i]['type_label']
            type_label.append(type)
        if with_cls:
            cls = imdb[i]['label']
            cls_label.append(cls)
        if with_bbox:
            bbox_target = imdb[i]['bbox_target']
            bbox_reg_target.append(bbox_target)

        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = image_processing.transform(im,True)
        processed_ims.append(im_tensor)



    return processed_ims, cls_label, type_label, bbox_reg_target

def get_minibatch(imdb, num_classes, im_size, with_type, with_cls, with_bbox, thread_num = 4):
    # im_size: 12, 24 or 48
    #flag = np.random.randint(3,size=1)
    num_images = len(imdb)
    thread_num = max(2,thread_num)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    #print(num_per_thread)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb,num_classes,im_size,with_type, with_cls, with_bbox))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    if with_type:
        type_label = list()
    if with_cls:
        cls_label = list()
    if with_bbox:
        bbox_reg_target = list()
    
    for t in range(thread_num):
        cur_process_ims, cur_cls_label, cur_type_label, cur_bbox_reg_target = threads[t].get_result()
        processed_ims = processed_ims + cur_process_ims
        if with_type:
            type_label = type_label + cur_type_label
        if with_cls:
            cls_label = cls_label + cur_cls_label
        if with_bbox:
            bbox_reg_target = bbox_reg_target + cur_bbox_reg_target
    
    im_array = np.vstack(processed_ims)
    if with_type:
        type_label_array = np.array(type_label)
    if with_cls:
        label_array = np.array(cls_label)
    if with_bbox:
        bbox_target_array = np.vstack(bbox_reg_target)
    
    data = {'data': im_array}
    label = {}
    if with_type:
        label['type_label'] = type_label_array
    if with_cls:
        label['label'] = label_array
    if with_bbox:
        label['bbox_target'] = bbox_target_array
    
    return data, label

def get_testbatch(imdb):
    assert len(imdb) == 1, "Single batch only"
    filename = imdb[0]['image']
    im = cv2.imread(filename)
    #print filename
    im_array = im
    data = {'data': im_array}
    label = {}
    return data, label
