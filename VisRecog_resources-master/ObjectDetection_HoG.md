
## Object Detection (Human Detection) using HoG (Dalal and Triggs)

### [Link To CVPR'05 Dalal & Triggs Paper](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

### [A tutorial from a blog](http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/)

### Short Summary (Add in a way to keep it concise and insightful)

The method is quite similar to SIFT. However it is important to note differences:

- It is a robust dense coding of spatial form , instead of a Sparse representation of an image as in SIFT.
- There is no dominant orientation alignment as in SIFT.
- SIFT descriptors are meant to be used individually and associated with a Keypoint, while HoG descriptors are associated with a block.
- It uses a “global” feature to describe a person rather than a collection of “local” features. Put simply, this means that the entire person is represented by a single feature vector, as opposed to many feature vectors representing smaller parts of the person. 
- Normalization is done over the block, to reduce the effects of gradient changes due to illumination etc and keep it robust. Finally Linear or Kernel SVM is used to classify results in positive(Human) or Negative Categories.

---

### Hard negative mining (from [reddit](https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/?ref=share&ref_source=link))
- Let's say I give you a bunch of images that contain one or more people, and I give you bounding boxes for each one. Your classifier will need both positive training examples (person) and negative training examples (not person). For each person, you create a positive training example by looking inside that bounding box. But how do you create useful negative examples?
- A good way to start is to generate a bunch of random bounding boxes, and for each that doesn't overlap with any of your positives, keep that new box as a negative. Ok, so you have positives and negatives, so you train a classifier, and to test it out, you run it on your training images again with a sliding window. But it turns out that your classifier isn't very good, because it throws a bunch of false positives (people detected where there aren't actually people).
- A hard negative is when you take that falsely detected patch, and explicitly create a negative example out of that patch, and add that negative to your training set. When you retrain your classifier, it should perform better with this extra knowledge, and not make as many false positives.

---

### Doubts

- 1. (Asim) Section 6.3 in above Paper 
- "To reduce aliasing, votes are interpolated bilinearly between the neighbouring bin centres in both orientation and position" What is the meant by above?


