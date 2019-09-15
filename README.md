# GeometricML

Repo containing small ML projects for testing different ideas.  I'm interested in ML in 3D contexts and using CNN based architectures to infer pose from images of known primitives.

A good starting point would be determining pose from low resolution renders of Lego pieces from Blender.  That's what all these Python scripts do now (classification, rotation estimation), although the ultimate goal is to implement/train <a href = "https://arxiv.org/pdf/1711.00199.pdf">PoseCNN</a> to infer the 3D poses of all Lego pieces in a combination.  My effete MacBook Pro is no match for this task but with a proper PC I think it should be doable.

(End bad ideas)

I've since abandoned the rambo deep learning approach, PoseCNN was a bit too complex to really understand and demanded too much computationally.  For now, the segmentation problem-solving will be postponed.  Each category of piece tested will be limited to a particular color.  Keypointnet will be retrained on each piece type separately.  A lot of work still needs to be done in best determining piece groupings for keypointnet.
