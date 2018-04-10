This code is provided to help gather statistics on your results on the
changedetection.net dataset.

Call processFolder(datasetPath, binaryRootPath)
ex : processFolder('C:\dataset', 'C:\results')

	*** This code calculates 7 metrics, namely (1)Recall, (2)Specificity, (3)FPR, (4)FNR, (5)PBC, (6)Precision, (7)FMeasure. The metric "FPR-S" is only calcualated for "Shadow" category on the server side, but not in this code. If it's really necessary, 

FPR_S = float(nbShadowError) / nbShadow

where nbShadowError is the number in the last column in the 'cm' file you get, that is the number of times a pixel is labeled as shadow in GT but detected as moving object. nbShadow is the total number of pixel labeled as shadow in GT for a video or category.****

	***Please notice that in the metrics you calculate may different from the ones that are going to be shown on changedetection.net, since only half of the ground truth is available to calculate locally with this code, while the changedetection.net calculates metrics based on all the ground truth.***