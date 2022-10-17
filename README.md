# Jamet_Kd_NN
---
 An implementation of a neural net created by Cedric Jamet, for estimating the K<sub>d</sub> values at any wavelength. It is capable of producing K<sub>d</sub> in the visible and into the ultraviolet, using input R<sub>rs</sub> at MODIS wavelengths (443 nm, 488 nm, 531 nm, 547 nm, 667 nm). This software is intended to be incorporated into Loisel and Stramski 2 (LS2), and provides K<sub>d</sub> values. 

This README document provides infromation about the files within the Jamet_Kd_NN repository. The directories Functions, LUTs, and Testing on the top layer of the repository contain a files used to run and perform tests of the Kd_NN module, KdNN.m, which are described below:

---

## KdNN.m:
Functionalized neural net model to compute the K<sub>d</sub> values at wavelength inputs, for *m* samples of R<sub>rs</sub>(λ) and solar zenith angle, where λ is at MODIS wavelengths. Also takes in 1 or *m* target wavelengths (lam). Returns K<sub>d</sub>(lam). See supporting documentation for further details.

## Functions 
Directory of functions necessary to run KdNN.m.
>**MLP_Kd.m**: The main function to compute the neural network, takes weights, biases, and normalization factors and runs the inputs through the neural net architecture. See supporting documentation for further details.\
>\
>**Move_Lam.m**: A function to compute the R<sub>rs</sub> at MODIS wavelengths for a given set of R<sub>rs</sub>(λ) and λ's. Uses linear interpolation to find the values at the MODIS wavelengths. See supporting documentation for further details.

## LUTs
Directory of look-up tables (LUTs) necessary to run KdNN.m
>**NN_weights_clear.xlsx**: LUT of the weights and the biases for the neural net for clear water types. \
>\
>**NN_weights_turbid.xlsx**: LUT of the weights and the biases for the neural net for turbid water types.\
>\
>**TRAINING_DATA.xlsx**: LUT of the normalization means and standard deviations for MODIS-wavelength R<sub>rs</sub>'s and K<sub>d</sub>'s, provided by our collaborators. The mean and standard deviation values of the training data.

## Testing 
Directory of functions and data used to test and evaluate the performance of KdNN.m against measured data provided to us by our collaborators and an RST data set. The functions within this folder do not need to be in your path to run KdNN.m.
>**Testing_Script.m**: A function which runs tests of the K<sub>d</sub> and produces comparative outputs. 
>
>### Test_Data
>Directory of data for testing.
>>**RST_Data_Clean.mat**: Multispectral data of 244 samples containing R<sub>rs</sub>'s, wavelengths, solar zenith angles, and K<sub>d</sub>'s. Not at MODIS wavelengths, so the R<sub>rs</sub>'s must be calculated at MODIS wavelengths.\
>>\
>>**inputs_test.dat**: Testing data given to us by our collaborators, at MODIS wavelengths. Contains R<sub>rs</sub>, $\mu_w$, and target wavelengths for evaluating K<sub>d</sub>.\
>>\
>>**outputs_test.dat**: Testing data given to us by our collaborators, containing the K<sub>d</sub> values at the target wavelengths in **inputs_test.dat**. 
>>
>### Outputs
>Directory of testing outputs. 
>>**NN_testing_Jamet.pdf**: Image file comparing the true and estimated K<sub>d</sub> values given by **inputs_test.dat** and **outputs_test.dat**. Also includes the median and standard deviation of the absolute error. The line in the plot is $x=y$. \
>>\
>>**NN_testing_RST.pdf**: Image file comparing the true and estimated K<sub>d</sub> values given by **RST_Data_Clean.mat**. Also includes the median and standard deviation of the absolute error. Log-log scaled, with the line in the plot $x=y$. \
>>

---
Matthew Kehrli, Aster Taylor, Rick Reynolds & Dariusz Stramski
*mdkehrli@ucsd.edu | mdkehrli@gmail.com\
Ocean Optics Research Laboratory, Scripps Institiution of Oceanography, University of California San Diego
