# ALGORITHM (Abstract)
1) To detect the regions of characters from a natural image.

2) Then the detected regions are filtered reduce the false postives and then it is feed to the Neural Network which classifies each image to its corresponding characters.

3) Then these characters are then printed on the console.


# ALGORITHM (Detailed)
1) First, we apply the canny filed to highlight the edges of the natural scene.

2) The highlighted edges are then used to highlight the contours which are then extracted with the help of OpenCV.

3) The contours are then passed to the text detection algorithm ( SWT, MSER ,ER ) and then the if the contours are a text, they are marked. And to reduce the number of redundant number of detected regions overlapping regions are grouped together and merged.  

4) The marked masks are then feed to ConvNet. The Neural Network is trained with fonts and handwritted characters from the Chars74K dataset. Then this ConvNet is used to evaluate the mask and if the mask is classified into a letter, then the characters are printed in the console.

# Solutions Tried 
1) I tried SWT from scratch but the performance was too bad, so had to go for the vannila version of NM classifier used in OpenCV.

2) Implemented the Tessaract-OCR version for character identification to mesure the performance achieved so far, but in the next release will go with a full fledged Neural Network using TensorFlow.
 
# ALPHA Ver 0.2

1) I have used the Opencv implementation of the algorithm refered in the paper "Real-Time Scene Text Localization and Recognition" by Luka ́sˇ Neumann and  Jiˇr ́ı Matas, to detect regions of text present in a image.

2) Then the regions are extracted into a mask and then it is fed to the tessaract ocr for robust identification of alphanumerical charecters in the natural scene image.

3) The extracted characters are then displayed on the console. 
