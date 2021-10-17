import numpy as np
import scipy.misc 
import os
import itertools

def random_augment(img):    
    img = img.astype(np.float32) / 255    
    H,W,C = img.shape
    imgs_aug = np.zeros((50,H,W,C))        
    gam_vals = np.random.uniform(0.8, 1.2, 7) # gamma
    con_vals = np.random.uniform(0.5, 1.5, 7) # contrast
        
    val_list = list(itertools.product(gam_vals, con_vals))    
    count = 0
    for tup_pair in val_list:
        gm = tup_pair[0]
        cn = tup_pair[1]
        tmpimg = img ** gm
        tmpimg = tmpimg * cn
        imgs_aug[count,:,:,:] = tmpimg.clip(0,1)
        count+=1    
    # add assertion that count == 49 (7*7)
    imgs_aug[count:,:,:] = img #original image at 50th pos
    
    return imgs_aug

def save_aug_imgs(imgPath, folder, imgName):
    img = scipy.misc.imread(imgPath)
    aug_imgs = random_augment(img)    
    N = aug_imgs.shape[0]
    fList = []
    for i in range(N):
        fname = folder + imgName + '_' + str(i) + '.png'
        scipy.misc.imsave(fname, aug_imgs[i,:,:,:])
        fList.append(fname)    
    return fList
    
'''
# CREATE FOLDER BEFORE RUNNING
import os
import imgAug
imgn = 'test3'
imgPath = './' + imgn + '.png'
folder = os.path.join(os.getcwd(), imgn+'/')
l = imgAug.save_aug_imgs(imgPath, folder, imgn)
'''

