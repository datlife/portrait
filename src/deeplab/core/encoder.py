"""Implementation of DeepLabV3+  as encoder - A deep atrous separable 
convolution neural network. (so wordy!!!)

For object segmentation task:
    * Output stride (input / output res. ratio)  = 16 (or 8) 
    for denser feature extraction.

    * Atrous Conv Rate = 2 , or 4 to the last two blocks/
    (for output stride = 8).
"""

# Two options here: Xception (for server-side) or MobileNetV2 (for Mobile Apps)

