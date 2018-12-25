import mxnet as mx
import numpy as np
import minibatch

class TestLoader(mx.io.DataIter):
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)
        self.index = np.arange(self.size)

        self.cur = 0
        self.data = None
        self.label = None

        self.data_names = ['data']
        self.label_names = []

        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return []#[(k, v.shape) for k, v in zip(self.label_names, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = []
        for i in range(cur_from,cur_to):
            idx = self.index[i]
            imdb_ = dict()
            annotation = self.imdb[idx].strip().split(' ')
            imdb_['image'] = annotation[0]
            
            imdb.append(imdb_)
        #print imdb
        data, label = minibatch.get_testbatch(imdb)
        self.data = [mx.nd.array(data[name]) for name in self.data_names]
        #self.label = [mx.nd.array(label[name]) for name in self.label_names]

class ImageLoader(mx.io.DataIter):
    def __init__(self, imdb, im_size, with_cls, with_bbox, batch_size, thread_num, flip=True, shuffle=False, ctx=None, work_load_list=None):

        super(ImageLoader, self).__init__()

        self.imdb = imdb
        self.batch_size = batch_size
        self.thread_num = thread_num
        self.im_size = im_size
        self.with_cls = with_cls
        self.with_bbox = with_bbox
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        self.cur = 0
        self.image_num = len(imdb)
        if flip:
            self.size = self.image_num*2
        else:
            self.size = self.image_num
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None
		
        
        self.label_names= ['label', 'bbox_target']
        self.with_type = False
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [('data', self.data[0].shape)]
      #  return [(k, v.shape) for k, v in zip(self.data_name, self.data)]


    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]


    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        #print cur_from,cur_to,self.index[cur_from:cur_to]
        imdb = []
        for i in range(cur_from,cur_to):
            idx = self.index[i]
            imdb_ = dict()
            is_flip = False
            if idx >= self.image_num:
                imdb_['flipped'] = True
                is_flip = True
                idx = idx - self.image_num
            else:
                imdb_['flipped'] = False
				
            annotation = self.imdb[idx].strip().split(' ')
            imdb_['image'] = annotation[0]+'.jpg'
            #print(imdb_['image'])
            label = int(annotation[1])
            if self.with_type:
                imdb_['type_label'] = int(label)
               
            if label == 1: #pos
                if self.with_cls:
                    imdb_['label'] = 1
                if self.with_bbox:
                    bbox_target = np.array(annotation[2:],dtype=np.float32)
                    if is_flip:
                        bbox_target[0], bbox_target[2] = -bbox_target[2], -bbox_target[0]
                    imdb_['bbox_target'] = bbox_target
            elif label == 0:              #neg
                if self.with_cls:
                    imdb_['label'] = 0
                if self.with_bbox:
                    imdb_['bbox_target'] = np.zeros((4,))
            elif label == -1:
                if self.with_cls:
                    imdb_['label'] = -1
                if self.with_bbox:
                    bbox_target = np.array(annotation[2:],dtype=np.float32)
                    if is_flip:
                        bbox_target[0], bbox_target[2] = -bbox_target[2], -bbox_target[0]
                    imdb_['bbox_target'] = bbox_target
            elif label == -2:             #landmark
                if self.with_cls:
                    imdb_['label'] = -1
                if self.with_bbox:
                    imdb_['bbox_target'] = np.zeros((4,))

            imdb.append(imdb_)
        
        data, label = minibatch.get_minibatch(imdb, self.num_classes, self.im_size, self.with_type, self.with_cls, self.with_bbox, self.thread_num)
        self.data = [mx.nd.array(data['data'])]
        self.label = [mx.nd.array(label[name]) for name in self.label_names]
