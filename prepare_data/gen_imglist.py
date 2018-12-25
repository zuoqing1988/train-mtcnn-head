import argparse
import numpy as np
import numpy.random as npr
import os,sys
sys.path.append(os.getcwd())
from config import config

def gen_imglist(size=20, base_num=1, with_hard=False, with_landmark=False):
    with open('%s/prepare_data/%d/part.txt'%(config.root, size), 'r') as f:
        part = f.readlines()
		
    with open('%s/prepare_data/%d/pos.txt'%(config.root, size), 'r') as f:
        pos = f.readlines()

    with open('%s/prepare_data/%d/neg.txt'%(config.root, size), 'r') as f:
        neg = f.readlines()

    if with_hard:
        with open('%s/prepare_data/%d/neg_hard.txt'%(config.root, size), 'r') as f:
            neg_hard = f.readlines()
    
    if with_landmark:
        with open('%s/prepare_data/%d/landmark.txt'%(config.root, size), 'r') as f:
            landmark = f.readlines()

    landmark_num = 0
    neg_hard_num = 0

    pos_num = base_num*300000 
    part_num = base_num*300000  
    if with_hard:
        neg_num = base_num*600000  
        neg_hard_num = base_num*300000 
    else:
        neg_num = base_num*900000
    if with_landmark:
        landmark_num = base_num*600000

    if with_hard:
        if with_landmark:
            out_file = "%s/prepare_data/%d/train_%d_with_hard_landmark_%d.txt"%(config.root, size, size, base_num)
        else:
            out_file = "%s/prepare_data/%d/train_%d_with_hard_%d.txt"%(config.root, size, size, base_num)
    else:
        if with_landmark:
            out_file = "%s/prepare_data/%d/train_%d_with_landmark_%d.txt"%(config.root, size, size, base_num)
        else:
            out_file = "%s/prepare_data/%d/train_%d_%d.txt"%(config.root, size, size, base_num)
    with open(out_file, "w") as f:
        if len(pos) > pos_num:
            pos_keep = npr.choice(len(pos), size=pos_num, replace=False)
            print('pos_num=%d'%pos_num)
            for i in pos_keep:
                f.write(pos[i])			
        else:
            print('pos_num=%d'%len(pos))
            f.writelines(pos)
        if len(part) > part_num:
            part_keep = npr.choice(len(part), size=part_num, replace=False)
            print('part_num=%d'%part_num)
            for i in part_keep:
                f.write(part[i])			
        else:
            print('part_num=%d'%len(part))
            f.writelines(part)
        if len(neg) > neg_num:
            neg_keep = npr.choice(len(neg), size=neg_num, replace=False)
            print('neg_num=%d'%neg_num)
            for i in neg_keep:
                f.write(neg[i])			
        else:
            print('neg_num=%d'%len(neg))
            f.writelines(neg)

        if with_hard:
            if len(neg_hard) > neg_hard_num:
                neg_hard_keep = npr.choice(len(neg_hard), size=neg_hard_num, replace=False)
                print('neg_hard_num=%d'%neg_hard_num)
                for i in neg_hard_keep:
                    f.write(neg_hard[i])			
            else:
                print('neg_hard_num=%d'%len(neg_hard))
                f.writelines(neg_hard)
        
        if with_landmark:
            if len(landmark) > landmark_num:
                landmark_keep = npr.choice(len(landmark), size=landmark_num, replace=False)
                print('landmark_num=%d'%landmark_num)
                for i in landmark_keep:
                    f.write(landmark[i])			
            else:
                print('landmark_num=%d'%len(landmark))
                f.writelines(landmark)


def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', dest='size', help='20 or 24 or 48', default='20', type=str)
    parser.add_argument('--base_num', dest='base_num', help='base num', default='1', type=str)
    parser.add_argument('--with_hard', dest='with_hard', help='with_hard', action='store_true')
    parser.add_argument('--with_landmark', dest='with_landmark', help='with_landmark', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    gen_imglist(int(args.size), int(args.base_num), args.with_hard, args.with_landmark)