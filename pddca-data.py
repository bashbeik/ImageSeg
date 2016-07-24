import nrrd
from os import *
import numpy as np
import lmdb
import scipy.misc

classes = {'BrainStem.nrrd' : 1,
           'Chiasm.nrrd' : 2,
           'Mandible.nrrd' : 3,
           'OpticNerve_L.nrrd' : 4,
           'OpticNerve_R.nrrd' : 5,
           'Parotid_L.nrrd' : 6,
           'Parotid_R.nrrd' : 7,
           'Submandibular_L.nrrd' : 8,
           'Submandibular_R.nrrd' : 9}

def load_stack(spath):
    global cntr
    print(spath)
    imgdata,volinfo=nrrd.read(path.join(spath,'img.nrrd'))
    imgdata = (np.array(imgdata) + 1024)/16
    structures = listdir(path.join(spath,'structures'))
    labls = np.zeros_like(imgdata, dtype=np.uint8)
    for s in structures:
        labl = classes[s]
        print(labl)
        segdata,segvolinfo = nrrd.read(path.join(spath,'structures',s))
        labls = np.maximum(labls,labl*(np.array(segdata,dtype=np.uint8)))
    #print(np.min(imgdata))
    #print(np.max(imgdata))
    for z in range(0,imgdata.shape[2]):
        img_slice  = imgdata[:,:,z]
        labl_slice = labls[:,:,z]
        idx = "{:08}".format(cntr)
        cntr = cntr + 1
        scipy.misc.imsave(path.join('../data','out_images',idx)+'.png',img_slice)
        scipy.misc.imsave(path.join('../data','out_labels',idx)+'.png',labl_slice)


data_path = '../data/PDDCA-1.4.1_part1'
stack_paths = listdir(data_path)
cntr = 0
for sp in stack_paths:
    load_stack(path.join(data_path, sp))
