%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test script for the PACE Kd NN code. The Kd NN code is run for 1 specified
%input of multispectral remote-sensing reflectance (Rrs) and solar zenith
%angles (sza). The resulting output from the test script is saved to
%Kd_NN_test_run_PACE_YYYYMMDD.xls for comparison with the provided
%output file Kd_NN_test_run_PACE.xls
%
%References: 
%
%Jamet, C., H., Loisel and D., Dessailly (2012). Retrieval of
%the spectral diffuse attenuation coefficient Kd(l) in open and coastal
%ocean waters using a neural network inversion, Journal of Geophysical
%Research-Oceans, 117, C10023 (https://doi.org/10.1029/2012JC008076)
%
%Loisel, H., Stramski, D., Dessailly, D., Jamet, C., Li, L., & Reynolds, R.
%A. (2018). An Inverse Model for Estimating the Optical Absorption and
%Backscattering Coefficients of Seawater From Remote‐Sensing Reflectance
%Over a Broad Range of Oceanic and Coastal Marine Environments. Journal of
%Geophysical Research. Oceans, 123(3), 2141–2171.
%https://doi.org/10.1002/2017JC013632
%
%Jorge, D. S. F., Loisel, H., Jamet, C., Dessailly, D., Demaria, J.,
%Bricaud, A., Maritorena, S., Zhang, X., Antoine, D., Kutser, T., Bélanger,
%S., Brando, V. O., Werdell, J., Kwiatkowska, E., Mangin, A., & d’Andon, O.
%F. (2021). A three-step semi analytical algorithm (3SAA) for estimating
%inherent optical properties over oceanic, coastal, and inland waters from
%remote sensing reflectance. Remote Sensing of Environment, 263, 112537–.
%https://doi.org/10.1016/j.rse.2021.112537
%
%NOTE: this code has some modifications and enhancements compared to the
%original NN algorithm presented in Jamet et al. (2012)
%
%%Created: October 14, 2022
%Completed: October 29, 2022
%Updates: Septemeber 25, 2024 - New neural network developed by Daniel
%Schaffer for the PACE mission. Updated test version of code to 1 example
%input with 12 Rrs bands.
%
%M. Kehrli, R. A. Reynolds, and D. Stramski 
%Ocean Optics Research Laboratory, Scripps Institution of Oceanography
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear command window and workspace; close figures
clc; clearvars; close all;

%define input parameters:

%input Rrs [sr^-1]
Rrs = [0.00770118082908374 0.00653225606789238 0.00545938537052318 ...
    0.00329481470482208	0.00231947867377767	0.00139863360138718 ...
    0.000850923468223990 0.000311991829840375 0.000200481735885968 ...
    0.000165554907438621 8.85566531228137e-05 5.41054713452636e-05];

%input solar zenith angle [deg]
sza = 19.3902713100000;

%output wavelengths [nm]
lambda = 490;

%input Kd NN LUTs
load 'Kd_NN_LUT_PACE.mat'

%preallocate Kd
Kd = nan(size(lambda));

%calculate Kd using NN
for i = 1:numel(lambda)
    Kd(i)=Kd_NN_PACE(Rrs(i,:),sza(i),lambda(i),Kd_NN_LUT_PACE);
end

%save inputs and outputs into an excel file
T = table(Rrs(:,1),Rrs(:,2),Rrs(:,3),Rrs(:,4),Rrs(:,5),sza,lambda,Kd);
T.Properties.VariableNames = {'Input Rrs(443) [1/sr]',...
    'Input Rrs(488) [1/sr]','Input Rrs(531) [1/sr]',...
    'Input Rrs(547) [1/sr]','Input Rrs(667) [1/sr]','Input sza [deg]',...
    'Output Wavelength [nm]','Output Kd [1/m]'};
FormatOut = 'yyyymmdd';
outfile=[cd '\Kd_NN_test_run_PACE_' datestr(datetime,FormatOut)];
writetable(T,outfile,'FileType','spreadsheet','Sheet','Kd_NN_PACE_Output')
