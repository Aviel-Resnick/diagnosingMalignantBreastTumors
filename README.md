# Diagnosing Breast Cancer through Machine Learning

*Received a perfect score from the Pennsylvania Junior Academy of Science - 2018*

*Scored as First Place in C500 (Computer Science) at MontCo Science Fair - 2018*

*Received the Villanova Award for Applied Statistical Analysis* - 2018

# Abstract

Currently, the most common diagnostic method for breast cancer is a mammogram, essentially an x-ray of compacted breast mass, which yields an 87% successful diagnostic rate. This focus of this study was utilizing machine learning classifiers in order to develop a new method for the diagnosis of malignant breast tumors. An initial test of a variety of linear and nonlinear classifiers trained on the UCI "Breast Cancer Wisconsin (Diagnostic)" Data Set, and tested using 10-fold cross validation, was used to identify linear discriminant analysis as the optimal classifier for the diagnostic model.  It was then retrained, and applied to new data (a dedicated 20% of the dataset mentioned above). The final product was a complete diagnostic system which produced a 92.11% successful diagnostic rate, over 5% greater than mammograms, leading to the acceptance of the hypothesis, and the rejection of the null. While the results indicate a working model for the diagnosis of a malignant breast tumor, they also reveal important information about data from fine needle aspirates, such as their tendency to be split linearly. In addition, this model acts as a proof of concept for the application of machine learning and other forms of unsupervised computation for the diagnosis of diseases.

The dataset, specifically the “Breast Cancer Wisconsin (Diagnostic) Data Set” is computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image, including:

1. Radius (mean of distances from center to points on the perimeter) 
2. Texture (standard deviation of gray-scale values) 
3. Perimeter 
4. Area 
5. Smoothness (local variation in radius lengths) 
6. Compactness (perimeter^2 / area - 1.0) 
7. Concavity (severity of concave portions of the contour) 
8. Concave points (number of concave portions of the contour) 
9. Symmetry 
10. Fractal Dimension ("coastline approximation" - 1)

Overall, the results of this projects will indicate whether using machine learning has diagnostic benefits, and how they compare to modern diagnostic methods.

LOG:
https://docs.google.com/a/stu.lmtsd.org/document/d/e/2PACX-1vSTyvUhRx6OiNr_k_AsAYtF7HUkDvkGlVM9Is-20DeuOQeAj_RsbD1UDoGkDP0cgI1Am1nowkMVIzQe/pub
