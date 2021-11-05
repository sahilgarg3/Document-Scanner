# Document-Scanner
Scanning documents out of the image using OpenCV or we can say extraction of the document from the image
## [Document Scanner](Document_Scanner.py)

## Requirements
- OpenCV
- NumPy
- [utlis.py](utlis.py) which is the python file consisting multiple functions to perform specific task which could be required in many such project again and again.
---
## User Inputs
1. Height of the of the final output Image.
2. Width of the of the final output Image.
---
Inputs of **get_countour** function are as follows:
- Image
- Canny Image
- Filters (Minimum number of boundaries)
- Draw (Whether to draw the contours or not)
- MinArea (Minimum area of the contour to be considered)

Outputs of **get_countour** function are as follows:
- List
  - Contours
  - Area of the contours
  - Perimeter of the countours 
  - Number of boundaries
  - Center of the contour
  - Bounding Box 
- Image with Contours drawn or not depending upon the parameter
Note: Contours are in decreasing order w.r.t the area covered by the contour.
---
Inputs of **reorder** function is points.

This function is to reorder the given points in proper order to get points for Warp Perspective.

---
Inputs of **get_warp** function are as follows:
- Image
- Width and Height
- Points
- Final Width and Height
This function gives the warped image of the input image within given data points
---
Inputs of **concat** function are as follows:
- Scale (Scale of the original image)
- List of Images/Videos.

This funciton gives the concatinated window of the images/videos given as input to it in the list, irrespective of the nature, scale, dimensions of the image/video. One can have the desired matrix of the concat window, for that you need to give the input the list of the images in that nested or matrix format.

**Also, If the length of the nested lists are not same, then the nested list with maximum length would be the length of the final output image and the lists with less number of image/video element would be replaced by blanked image**
