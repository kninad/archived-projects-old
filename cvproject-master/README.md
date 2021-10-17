# cvproject
Tweaking the implementation of stereo based monocular depth prediction model

Major points are:

- No true depth data is required for training.
- Only stereo pairs of scenes are required.
- The network learns to predict the ground truth disparity and hence as a consequence, the depth. 
- Based on a cvpr'17 paper by cv group of ucl (clement godard et al.)

Found some inconsistency in computation of image gradient in the original code. [Link](https://github.com/mrharicot/monodepth/issues/46) to the github issue on the original repository. But it did not seem to have a major effect on the results.

The project report is available [here](https://ninception.github.io/docs/CV670_finalReport.pdf).
