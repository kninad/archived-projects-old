
##[Link To Paper](https://arxiv.org/pdf/1311.2524.pdf)

##PipeLine
1. Input Image
2. Extract Region Proposals
3. Warp those images and feed them to a CNN fine tuned for classification and obtain object annotations with locations.

Extraction of Region Proposals is a crucial step. Multiple methods are used, such as, selective search,
category-independent object proposals, constrained parametric min-cuts, multi-scale combinatorial grouping.

One Region Proposal method is Selective search, which is able to provide a hierarchical representation of image and generate 
segmentations hierarchically. 
Please Look at this [Stanford Slide for details](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/ssearch_schuyler.pdf).

You may also look at the [research paper](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf).

For a good succinct summary of the paper, please look at this [gitbooks](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/object_localization_and_detection.html) page. For information on 
**bounding box regression for object localization**, you can also look at the gitbooks link.

- [Nice slides for understanding the bounding box regression terms](http://cvlab.postech.ac.kr/~bhhan/class/cse703r_2016s/csed703r_lecture6.pdf)
 
 - What we are learning is the transformation of the proposal bounding box we are feeding to the CNN to the 'ground truth' bounding box. (the proposal comes from methods like Selective search etc..)

---

## SSD resources

- [ECCV 2016 slides](http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf)
- [slideshare tutorial](https://www.slideshare.net/xavigiro/ssd-single-shot-multibox-detector)
