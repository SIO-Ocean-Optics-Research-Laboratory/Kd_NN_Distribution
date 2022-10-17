%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB CODE FOR THE ESTIMATION OF THE DIFFUSE    
% ATTENUATION COEFFICIENT FROM RRS SPECTRUM        
%
% Running code to check the accuracy of the KdNN code. 
% References:
% Jamet, C., H., Loisel and D., Dessailly (2012). Retrieval of the spectral diffuse attenuation coefficient Kd(l) in open and coastal ocean waters using a neural network inversion, Journal of Geophysical Research-Oceans, 117, C10023, doi:10.1029/2012JC008076.
    % Loisel, H., D., Stramski, D., Dessailly, C., Jamet, L., Li, and R.A., Reynolds (2018). An inverse model for estimating the optical absorption and backscattering coefficients of seawater from remote-sensing reflectance over a broad range of oceanic and coastal marine environments. Journal of  Geophysical Research: Oceans, 123, https://doi.org/10.1002/2017JC013632    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

%%% Testing the NN on given data 
inputs=load('inputs_test.dat');
muw=inputs(:,7);
sza=asind(1.34*sind(acosd(muw)));

input_Kd_NN_LUT = load('Kd_NN_LUT.mat');
input_Kd_NN_LUT = input_Kd_NN_LUT.Kd_NN_LUT;

Kd_est=Kd_NN(inputs(:,1:5),sza,inputs(:,6),input_Kd_NN_LUT);

outputs=load('outputs_test.dat');

hold on;
scatter(outputs,Kd_est,'k.');
x=linspace(0,max([Kd_est;outputs]));
plot(x,x,'r');
title("True vs Estimated K_ds");
xlabel("IOCCG K_d [m^{-1}]")
ylabel("Estimated K_d [m^{-1}]")
text(0.25,4,strcat("Absolute Error Median: ",string(median(abs(Kd_est-outputs)))));
text(0.25,3.75,strcat("Absolute Error STD: ",string(std(abs(Kd_est-outputs)))));
%set(gca,'YScale','log');
%set(gca,'XScale','log');