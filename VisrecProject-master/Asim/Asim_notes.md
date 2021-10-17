Target - Obtain a high feature depth map of an indoor/outdoor scene which can help us understand the 3D 
state of the image environment.

Shortcomings of methods-
1. Far Off distance depth maps are poorly predicted.
2. Sharp boundaries are not visible in predicted depth maps.

We need Sharp Segmentation masks with fine object boundaries. Look at FB Deep Mask for this.

## Shadows are a major cue in depth perception.
## ViewPoint information can also be a cue. Figure out how to use viewpoint information.

Using Deep Mask and Sharp Mask we will be able to obtain high-fidelity boundaries with sharp delineations. 

This first level of segmentation will give us semantic objects in an image. This semantic segmentation will be useful in 
depth prediction as we want to improve the predicted depth around objects, and make boundaries of objects perceptible from 
depth images too.

However we still need to explicitly define/realize exactly what kind of segmentation will help us to predict accurate depth 
maps. After the first level semantic segmentation we can go down this path of "geometrical segmentation" for improving the 
"featureness" of the predicted depth map.

One possibility is:
1. Inspite of semantic segmentation, we look at segmenting objects with geometrical "connectivity".(Quite Vague as of now) (There is still a 
corelation with semantic segmentation, but there is a difference due to the data driven definition of semantic classes!).
(There must be a classical/modern work on this problem. Look at Geometry Processing community for leads).
-
* http://digitalassets.lib.berkeley.edu/techreports/ucb/text/EECS-2013-132.pdf
* http://www.cc.ac.cn/2011researchreport/201102.pdf
* http://www.cs.unc.edu/techreports/92-039.pdf

*http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.323.9299&rep=rep1&type=pdf * (Look at this paper)


## Realise it is not possible to obtain accurate distances in the depth maps with a monocular image. 

Formalise the iterative improvement scheme. 
Also try to formalise the idea of "depth contours" of the image.
It is

(Below Ideas are trying to utilize the frontiers of deep Learning. However our main aim should be able to integrate a Hierarchical Segmentation approach for this task.)
## Look at GAN for this task too.
## Look at ResNets too.



---

## Approaches

*Few approaches which need to be thought about:*

    Sliding Window Approach

    Hierarchical Segmentation Approach

** Multi scale deep netowrks , couple them with multi scale (depth - edge) information



Try to formalise the idea of (Gaussian) Ratios and try to include it in a loss function.





