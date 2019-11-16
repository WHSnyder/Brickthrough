#  Lego Photogrammeter

A three stage mess of a project for generating 3D Lego models from images.  The first stage uses a retrained Mask R-CNN model to extract 
masks for each piece in the image. The second stage uses known 2D-3D correspondences with a pnp solver (OpenCV) to estimate poses for each piece.  The third stage refines the pose estimates.  

The particular use case I had in mind was building some kind of simple fighter craft and flying the generated 3D model ingame.  The four pieces this primitve version uses are various x by x bricks, left-right versions of a basic wing piece, an engine type piece, and a pole for weapons.

See this <a href="https://github.com/WHSnyder/LegoTrainingRenderer">repo</a> for training data details...


## Challenges

Far too many to list here when it comes to my lack of general CV/DL experience and trouble digesting research papers... 

* First stage  

![alt text](./repo_images/tests.gif "inputs")  ![alt text](./repo_images/gts.gif "gts")  ![alt text](./repo_images/preds.gif "preds")  
	
	* Training data has frequent abnormalities such as pieces lying half off-screen or extreme occlusion.  In such cases a wing could be hidden entirely with the exception of a 2x2 region of studs.  If this wing is viewed from directly above, it will be indistinguishable from a 2x2 brick piece. The network will be forced to make the wrong choice given the infomation at hand.
	* The network is relatively accurate with real images of pieces, though struggles consistently with black pieces and differing resolutions.

* Second stage

	* Bad masks from stage 1 often make this stage hopeless.
	* More comming soon... 

* Third stage 

	* This is by far the easiest stage to execute considering 3D context is available and no complex deep learning concepts are required.  Better approaches are obviously needed but as of now studs and insets are brute-force matched with one another to find likely fits between nearby pieces.  Will fail if 2nd stage estimates are bad enough.


## Future directions

* After spending most of October surveying papers on SotA pose estimation algorithms/networks I came to the conclusion that estimating explicit 2D-3D correspondence would be the most realistic approach.  Fully deep-learned pose estimation pipelines such as OcclusionNet or PoseCNN were far too bulky/confusing to work with or understand and lighter networks such as KeypointNet didn't seem robust enough to heavy occlusion/textureless surfaces.  In the near future I would love to experiment with estimating dense correspondence with some kind of hierarchical FCN approach or a graph-based model as seen in OcclusionNet.  

* Training a depth map estimator.

* Exploring structure-from-motion techniques and possible integration with deep learning strategies. 

* Methods involving a guess-rerender-refine approach to pose estimation or an iterative method as in this <a href="https://arxiv.org/pdf/1507.06550.pdf">human pose estimation method.</a>   

* I don't any formal training or coursework in machine learning for computer vision so I could be wrong, but from my research it seems like a fully deep-learned pipeline encompassing all 3 stages would be impractical.  The need for fuzzy logic followed by structured 3D reasoning suggests a very messy soup of fully-connected layers and slow iteration time.  Pieces can be completely different or could have modular similarity to one another ie: 2 2x2 bricks make a 4x2 brick.

* Learning about deep learning more deeply.




