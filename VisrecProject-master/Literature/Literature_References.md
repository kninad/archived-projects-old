# Monocular (single-view) depth estimation

## David Eigen, Rob Fergus : Multi scale and 3 tasks at the same time

[Depth Map Prediction from a Single Image using a Multi-Scale Deep Network -- NIPS 2014](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network)
  
  - [NYU depth project page (Eigen-nips14)](https://www.cs.nyu.edu/~deigen/depth/)
  - [Code link -- Multi Scale , NIPS-14](https://github.com/hjimce/Depth-Map-Prediction)


*follow-up paper* 
[Predicting Depth, Surface Normals and Semantic Labels With a Common Multi-Scale Convolutional Architecture -- ICCV 2015](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.html)

  - [Project page: Predicting Depth, Surface Normals and Semantic Labels](http://www.cs.nyu.edu/~deigen/dnl/)
  - *Also has links to code, trained models and other stuff*

---

## State of the art : In the Wild

[Single-Image Depth Perception in the Wild -- NIPS 16](https://papers.nips.cc/paper/6489-single-image-depth-perception-in-the-wild)
  - Has a huge annotated dataset to be used. Images are annotated with relative depths between random points. So one experiment that we can try out is to try to give weakly annotated (noisy) results for depth (given by some other FAST method) as the Ground truth for the method described in this paper -- just like our paper review assignment.

  - [Project page: Depth Perception in the Wild ](http://www-personal.umich.edu/~wfchen/depth-in-the-wild/)
  - [Code link -- wild nips2016](https://github.com/wfchen-umich/relative_depth)

*related works to be seen*

[Zoran- Learning Ordinal Relationships for Mid-Level Vision](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zoran_Learning_Ordinal_Relationships_ICCV_2015_paper.html)

[Task Specific Edge Detection -- CVPR 16](https://arxiv.org/pdf/1511.03328.pdf)

[Which Edges Matter](http://www.cs.cmu.edu/~aayushb/pubs/edges.pdf)

---

*This work can be surveyed in the end*
## Fayao Liu, Chunhua Shen, Guosheng Lin

[Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields](http://ieeexplore.ieee.org/abstract/document/7346484/)

[Deep Convolutional Neural Fields for Depth Estimation From a Single Image](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Liu_Deep_Convolutional_Neural_2015_CVPR_paper.html)

[Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation - cvpr2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Lin_Efficient_Piecewise_Training_CVPR_2016_paper.html)

---

## Classical Works

[Project Page: Cornell, Saxena et al](http://www.cs.cornell.edu/~asaxena/learningdepth/)
  
  - Learning Depth from Single Monocular Images -- This Cornell project page has several papers related to this topic.
  - Also contains some results
  - Classical work can (maybe?) provide some important cues.

---

## Others

[Semi-Supervised Deep Learning for Monocular Depth Map Prediction -- arxiv17](https://arxiv.org/abs/1702.02706)

[Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue -- ECCV 2016](http://link.springer.com/chapter/10.1007/978-3-319-46484-8_45)

[Monocular Depth Estimation of Outdoor Scenes Using RGB-D Datasets -- ACCV 2016](http://link.springer.com/chapter/10.1007/978-3-319-54427-4_7)

[Occlusion Edges](https://arxiv.org/pdf/1412.7007.pdf)

*Misc*

- [Unsupervised Monocular Depth Estimation with Left-Right Consistency -- arxiv16](https://arxiv.org/abs/1609.03677)

- [Pulling Things out of Perspective](http://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Ladicky_Pulling_Things_out_2014_CVPR_paper.html)

- [Dense Monocular Depth Estimation in Complex Dynamic Scenes](https://web.stanford.edu/~cqf/papers/Dense_Monocular_Depth_CVPR2016.pdf)

- [(old)Learning Depth from Single Monocular Images](https://papers.nips.cc/paper/2921-learning-depth-from-single-monocular-images)

---

## Segmentation and Depth Estimation (later/abandon)

[Wang_Towards_Unified_Depth_2015_CVPR](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Towards_Unified_Depth_2015_CVPR_paper.pdf)

[Exploiting Depth From Single Monocular Images for Object Detection and Semantic Segmentation](http://ieeexplore.ieee.org/abstract/document/7707416/)

[Analyzing Modular CNN Architectures for Joint Depth Prediction and Semantic Segmentation](https://arxiv.org/abs/1702.08009)

[Hierarchical Image Segmentation based on Observation Scale](https://hal.archives-ouvertes.fr/hal-00789387/document)

[Occlusion Cues for Depth](http://download.springer.com/static/pdf/12/chp%253A10.1007%252F978-3-642-04617-9_13.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F978-3-642-04617-9_13&token2=exp=1490809990~acl=%2Fstatic%2Fpdf%2F12%2Fchp%25253A10.1007%25252F978-3-642-04617-9_13.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fchapter%252F10.1007%252F978-3-642-04617-9_13*~hmac=e0f715c8f45e18479ffa0e3197659d087d12eb0ce33cc8532763c63b9038ae55)


---

# Codes

[Saxena Ng et. al](http://make3d.cs.cornell.edu/code_linux.html)

