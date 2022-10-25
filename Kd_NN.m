function [Kd_est]=Kd_NN(Rrs,sza,lambda,Kd_LUT)
%Implements the neural network (NN) algorithm to calculate the diffuse 
%attenuation coefficient of downwelling planar irradiance (Kd) at one 
%preselected output light wavelength (lambda) using input remote-sensing 
%reflectance (Rrs) at MODIS wavelengths and solar zenith angle (sza)
%
%Reference: Jamet, C., H., Loisel and D., Dessailly (2012). Retrieval of
%the spectral diffuse attenuation coefficient Kd(l) in open and coastal
%ocean waters using a neural network inversion, Journal of Geophysical
%Research-Oceans, 117, C10023 (https://doi.org/10.1029/2012JC008076)
% 
%NOTE: this code has some modifications and enhancements compared to the
%%original NN algorithm presented in this 2012 paper
%
%Required function inputs: R_rs, sza, lambda, Kd_LUT
%   Rrs [nsamplx5 Double]: Values of spectral remote-sensing reflectance
%   [sr^-1] at MODIS light wavelengths: 443, 488, 531, 547, 667 [nm].
%   nsampl is the number of "sample" inputs, each consisting of five R_rs
%   values. Note: nsamp can be any positive integer, i.e., 1, 2, 3, etc.
%
%   sza [nsampx1 Double]: Solar zenith angle [deg] for each "sample" input
%   of Rrs values. As a result, one single complete "sample" input to the
%   NN algorithm consists of five Rrs values, i.e. Rrs(443), Rrs(488),
%   Rrs(531), Rrs(546), Rrs(667), and one value of solar zenith angle, sza.
%
%   lambda [nsamplx1 Double]: Output light wavelength [nm] at which the
%   desired value of Kd is estimated for a given "sample" input. Lambda
%   serves as an input parameter for the Kd_NN function and is defined by
%   user. Note: if multiple "sample" inputs are used (i.e., if nsamp > 1)
%   the output wavelength lam can be either the same or can differ between
%   the "sample" inputs which depends on how lam(nsampx1) is defined by
%   user. Note: light wavelengths are in vacuum
%
%   Kd_LUT [1x1 Structure]: Structure containing three required look-up
%   tables (LUTs); can be loaded via load('Kd_NN_LUT.mat')
%
%       Kd_LUT.weights_1: LUT with weights and biases from NN for case 1
%       waters
%
%       Kd_LUT.weights_2: LUT with weights and biases from NN for case 2
%       waters
%
%       Kd_LUT.train_switch: LUT with means and standard deviations of 
%       40,000 inputs and outputs used to train the NN
%
%Outputs: Kd_est
%   Kd_est (mx1 Double): The estimated value of the average diffuse
%   attenuation coefficient of downwelling planar irradiance [m^-1] between
%   the sea surface and first attenuation depth at preselected output light
%   wavelength (lambda) for each "sample" input of spectral Rrs and sza.
%
%Version History: 
%2018-04-04: Original implementation in C written by David Dessailly
%2020-03-23: Original Matlab version, D. Jorge 
%2022-09-01: Revised Matlab version, M. Kehrli
%DARIUSZ: WHEN WE ARE FINISHED, ADD DATE AND Final Revised Matab version: M. Kehrli, R. A. Reynolds and D. Stramski
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check function arguments and existence of LUTs
    arguments
        Rrs (:,5) double
        sza (:,1) double
        lambda (:,1) double
        Kd_LUT (1,1) struct
    end

    %number of samples
    nsamp = size(Rrs,1); 

    %copy input lam values into an array to match the number of samples if
    %only one input wavelength lambda is provided
    if length(lambda)==1
        lambda=repmat(lambda,nsamp,1);
    end

    %copy input sza values into an array to match the number of samples if
    %only one sza is provided
    if length(sza)==1
        sza=repmat(sza,nsamp,1);
    end

    %compute muw from sza
    muw = cosd(asind(sind(sza)/1.34));

    %if there are negative input Rrs, set to NaN
    Rrs(Rrs<0)=NaN;

    %combine inputs
    inputs = [Rrs,lambda,muw];

    %read in weights and biases for the NN
    weights_1 = Kd_LUT.weights_1; %case 1 weights
    weights_2 = Kd_LUT.weights_2; %case 2 weights

    %build the neural nets, case 2 then case 1 waters

    %read in weights and biases for case 2 waters:
    %number of inputs
    ne = 7;
    %number of neurons on the first hidden layer
    nc1 = 9;
    %number of neurons on the second hidden layer
    nc2 = 6;
    %number of neurons on the output layer
    ns = 1;

    %read case 2 NN LUT as weight and bias matricies 
    b1_2 = weights_2.('b1'); b1_2(isnan(b1_2)) = [];
    b2_2 = weights_2.('b2'); b2_2(isnan(b2_2)) = [];
    bout_2 = weights_2.('bout'); bout_2(isnan(bout_2)) = [];

    w1_2 = weights_2.('w1'); w1_2(isnan(w1_2)) = []; w1_2=reshape(w1_2,nc1,ne);
    w2_2 = weights_2.('w2'); w2_2(isnan(w2_2)) = []; w2_2=reshape(w2_2,nc2,nc1);
    wout_2 = weights_2.('wout'); wout_2(isnan(wout_2)) = [];wout_2=reshape(wout_2,ns,nc2);

    %read in weights and biases for case 1 waters:
    %number of inputs
    ne = 6;
    %number of neurons on the first hidden layer
    nc1 = 8;
    %number of neurons on the second hidden layer
    nc2 = 6;
    %number of neurons on the output layer
    ns = 1;

    %read case 1 NN LUT as weight and bias matricies 
    b1_1 = weights_1.('b1'); b1_1(isnan(b1_1)) = [];
    b2_1 = weights_1.('b2'); b2_1(isnan(b2_1)) = [];
    bout_1 = weights_1.('bout'); bout_1(isnan(bout_1)) = [];

    w1_1 = weights_1.('w1'); w1_1(isnan(w1_1)) = []; w1_1 = reshape(w1_1,nc1,ne);
    w2_1 = weights_1.('w2');w2_1(isnan(w2_1)) = []; w2_1 = reshape(w2_1,nc2,nc1);
    wout_1 = weights_1.('wout'); wout_1(isnan(wout_1)) = []; wout_1 = reshape(wout_1,ns,nc2);

    %mean and standard deviation LUT for each NN from training dataset of
    %40,000 inputs to normalize input data
    train_switch = Kd_LUT.train_switch;
    mu_switch = train_switch.('MEAN')';
    std_switch = train_switch.('STD')';

    %take the relevant mu and std data for case 2 waters
    mu_2 = mu_switch(2:9);
    std_2 = std_switch(2:9);

    %for case 1 waters ignore the Rrs(667) data, so remove the 667 nm %band
    mu_1 = mu_switch([2:5,7:9]);
    std_1 = std_switch([2:5,7:9]);

    %calculate the reflectance band ratio of Rrs488/Rrs547 to classify
    %%input Rrs as case 1 or case 2 waters
    ratio = inputs(:,2)./inputs(:,4);

    %find which samples are case 1 or case 2 waters
    val_2 = find(ratio<0.85);
    val_1 = find(ratio>=0.85);

    %get case 1 and case 2 water type inputs
    x_case1 = inputs(val_1,[1:4,6:7]); 
    x_case2 = inputs(val_2,1:7);

    %preallocate for input normalization
    x_case2_N = ones(size(x_case2));
    x_case1_N = ones(size(x_case1));

    %normalize input data for case 1 and case 2 waters
    for j = 1:7 
        x_case2_N(:,j) = (2/3)*((x_case2(:,j)-mu_2(j))/std_2(j));
    end
    for j = 1:6
        x_case1_N(:,j) = (2/3)*((x_case1(:,j)-mu_1(j))/std_1(j));
    end

    %Kd inversion for case 2 waters
    [Kd_est_case2] = MLP_Kd(x_case2_N,w1_2,b1_2,w2_2,b2_2,wout_2,...
        bout_2,mu_2(8),std_2(8));

    %Kd inversion for case 1 waters
    [Kd_est_case1] = MLP_Kd(x_case1_N,w1_1,b1_1,w2_1,b2_1,wout_1,...
        bout_1,mu_1(7),std_1(7));

    %combine output into single variable
    Kd_est(val_1) = Kd_est_case1;
    Kd_est(val_2) = Kd_est_case2;
    Kd_est = Kd_est';

    %set all inputs which contain NaNs to output NaN
    Kd_est(isnan(sum(inputs,2))) = NaN;
end
%end of main code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Additional subfunctions that are called
function [Kd_est]=MLP_Kd(x,w1,b1,w2,b2,wout,bout,muKd,stdKd)
%Computes the NN output from the inputs. Assumes 2 hidden layers, and
%utilizes a tanh activation function. The inputs are fed into the first
%layer as a matrix, then the second layer, then are collapsed to an output
%value. This function works for hidden, output, and input layers of any
%size, so long as the weight and bias functions are the right size and
%there are 2 hidden layers.
%
%Also note that the output of the NN is assumed to be log10(Kd), and so it
%is raised to the power of 10 to ensure the correct output of Kd. Because
%Kd has a high dynamic range, the logarithm helps to ensure the efficacy of
%the NN. In addition, the output from the NN is assumed to be normalized
%for efficacy, so denormalization constants are added in to reverse this.
%
%Inputs: x, w1, b1, w2, b2, wout, bout, muKd, stdKd
%   x (nsamp x ni Double): The inputs to the NN, with the number of 
%	"sample" inputs, nsamp, and the number of input neurons, ni. Note 
%  	that six input neurons are used for case 1 waters and seven 
%	input neurons are used for case 2 waters
%		
%   w1 (nc1 x ni Double): Connection weights of the first hidden
%	layer, which has (nc1) neurons and connects to the (ni) neurons
%	in the input layer
%
%	b1 (nc1 x 1 Double): Neuron bias of the first hidden layer, 
%	which has (nc1) neurons
%
%	w2 (nc2 x nc1 Double): Connection weights of the second hidden
%	layer, which has (nc2) neurons. This connects to the (nc1) 
%   neurons in the first hidden layer
%
%	b2 (nc2 x 1 Double): Neuron bias of the second hidden layer,
%   which has (nc2) neurons
%       
%   wout (1 x nc2 Double): Connection weights of the output layer, 
%   which connects to the (nc2) neurons of the second hidden layer
%	and returns only a single output
%
%   bout (1x1 Double): Neuron bias of the output layer, which only 
%   has 1 neuron
%
%   muKd (1x1 Double): The mean output of Kd values from NN training, for
%   denormalization of the output
%
%   stdKd (1x1 Double): The std output of Kd values from NN training, for
%   denormalization of the output
%
%Outputs: Kd_est
%   Kd_est (nsamp x 1 Double): The estimated Kd values for the nsamp
%   "sample" inputs
%        
%
%Created: July 6, 2022
%Completed: July 12, 2022
%Updates: N/A
%
%Aster Taylor and Matthew Kehrli
%Ocean Optics Research Laboratory, Scripps Institution of Oceanography
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%get the number of rows/samples
[rx,~]=size(x);

%forward data through the NN:
%compute first hidden layer, using tanh activation function
a=1.715905*tanh(0.6666667*(x*w1'+(b1*ones(1,rx))'));
%compute second hidden layer, using tanh activation function
b=1.715905*tanh((2./3)*(a*w2'+(b2*ones(1,rx))'));
%compute final output 
y=b*wout'+bout*ones(rx,1);

%denormalize because NN is trained for normalized log-transformed data
Kd_est=10.^(1.5*y*stdKd+muKd);
end
