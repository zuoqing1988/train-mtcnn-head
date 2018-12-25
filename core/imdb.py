import os

class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, mode='train'):
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.mode = mode
        self.cache_path = os.path.join(self.root_path, 'cache')
        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.annotations = self.load_annotations()

    def get_annotations(self):
        return self.annotations

    def load_annotations(self):

        annotation_file = os.path.join(self.data_path, 'imglists', self.image_set + '.txt')
        assert os.path.exists(annotation_file), 'annotations not found at {}'.format(annotation_file)
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
        return annotations
		
    def write_results(self, all_boxes):
        """write results

        Parameters:
        ----------
        all_boxes: list of numpy.ndarray
            detection results
        Returns:
        -------
        """
        print 'Writing fddb results'
        res_folder = os.path.join(self.cache_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # save results to fddb format
        filename = os.path.join(res_folder, self.image_set + '-out.txt')
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(self.image_set_index):
                f.write('%s\n'%index)
                dets = all_boxes[im_ind]
                f.write('%d\n'%dets.shape[0])
                if len(dets) == 0:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.5f}\n'.
                            format(dets[k, 0], dets[k, 1], dets[k, 2]-dets[k, 0], dets[k, 3]-dets[k, 1], dets[k, 4]))