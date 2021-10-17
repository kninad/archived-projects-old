## Tasks

- Study Loss functions of Various Papers.
- Study works on Depth Edges (Occlusion Edges). For integrating them explicitly into the pipeline at multiple scales.
- Experiment: Loss function at the coarser level with respect to Blur of GT depth image.  
- Experiment: Try to find Depth/Occlusion edges from GT depth map.
- Identify the major contributions of each paper as in what they established firmly. Like the Wang et al paper established       that doing a joint learning of semantic class and depth image improves both of their performance. We want to have a list of "proved" conclusions of each paper. A type of broad survey of recent and state of the art works.

- Depth blurring

## Ideas

- Joint Learning has been shown to be a promising approach. We may probably need to redefine what needs to be jointly learnt.



---

# OLD

### Hypothesis: 
Depth Perception and Segmentation aid each other. 
For dense segmentation depth cues are important.
For dense hierarchical segmentation, depth cues may help to generate segmentation mask for segmenation at the next level.

### Aim: 
Try to design a cooperative procedure between depth prediction and image segmentation. End goal should be that predicted
depth should have high local variations(predicted depth should not be smooth at a location, if an object at that location is not 
smooth). The segmentation should be dense hierarchical segmentation.

### Approach:
1. Use multi scale windows in RGB image to enhance the depth prediction at a particular image patch.
Use cooperation between neighnouring windows for global continuity and smoothness. Using the changed depth prediction, for a 
particular foreground object, construct a hierarchical segmentation mask for the next level.



