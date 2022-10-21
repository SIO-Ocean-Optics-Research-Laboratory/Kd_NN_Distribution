function [Kd_est]=Kd_NN(Rrs,sza,lam,Kd_LUT)
%Takes as input Rrs at MODIS wavelengths, sza, and lam which defines output wavelengths for Kd. Using the 
%Jamet et al. neural network (NN) algorithm, returns the Kd values at output wavelengths given
%input Rrs and sza.
%
%Reference: Jamet, C., H., Loisel and D., Dessailly (2012). Retrieval of the
%spectral diffuse attenuation coefficient Kd(l) in open and coastal ocean
%waters using a neural network inversion, Journal of Geophysical
%Research-Oceans, 117, C10023 (https://doi.org/10.1029/2012JC008076).
%
%Required function inputs:
%   R_rs [mx5 Double]: Remote-sensing reflectance [sr^-1] at MODIS wavelengths 
%       (443, 488, 531, 547, 667 nm) 
%
%   sza [mx1 Double]: Solar zenith angle [deg] for each sample of input Rrs 
%
%   lam [mx1 Double]: Output wavelength [nm]; also an input parameter for
%   the neural network
%
%   Kd_LUT [1x1 Structure]: Structure containing three required look-up
%   tables; can be loaded via load('Kd_NN_LUT.mat')
%
%       Kd_LUT.weights_1: LUT of weights and biases from NN for case 1
%       waters
%
%       Kd_LUT.weights_2: LUT of weights and biases from NN for case 2
%       waters
%
%       Kd_LUT.train_switch: LUT of means and standard deviations of 40000
%       inputs and outputs used to train the neural net
%
%Outputs: Kd_est
%   Kd_est (mx1 Double): The estimated Kd values at each selected output wavelength lam
%   for each input sample of Rrs: Kd(lam) [m^-1]
% 
%Created: July 6, 2022
%Completed: July 12, 2022
%Updates: N/A
%
%Aster Taylor and Matthew Kehrli
%SIO Ocean Optics Research Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check function arguments and existence of LUTs
    arguments
        Rrs (:,5) double
        sza (:,1) double
        lam (:,1) double
        Kd_LUT (1,1) struct
    end

    %number of samples
    nsamp = size(Rrs,1); 

    %copy input wavelengths (lam) into an array to match the number of samples if
    %only one input wavelength is provided
    if length(lam)==1
        lam=repmat(lam,nsamp,1);
    end

    %copy input values of sza into an array to match the number of samples if
    %only one input wavelength is provided
    if length(sza)==1
        sza=repmat(sza,nsamp,1);
    end

    %compute muw from the sza
    muw = cosd(asind(sind(sza)/1.34));

    %if there are negative input Rrs, set to NaN
    Rrs(Rrs<0)=NaN;

    %combine inputs
    inputs = [Rrs,lam,muw];

    %read in weights and biases for the NN
    weights_1 = Kd_LUT.weights_1; %case 1 wieghts
    weights_2 = Kd_LUT.weights_2; %case 2 wieghts

    %Build the neural nets, case 2 then case 1 waters

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

    %mean and standard deviation LUT for each NN, from training dataset of
    %40,000 inputs to normalize input data
    train_switch = Kd_LUT.train_switch;
    mu_switch = train_switch.('MEAN')';
    std_switch = train_switch.('STD')';

    %take the relevant mu and std data for case 2 waters
    mu_2 = mu_switch(2:9);
    std_2 = std_switch(2:9);

    %for the case 1 water ignore the 667nm data, so remove that band
    mu_1 = mu_switch([2:5,7:9]);
    std_1 = std_switch([2:5,7:9]);

    %calculate the ratio of Rrs488/Rrs547 to classify input Rrs as case 1
    %or case 2
    ratio = inputs(:,2)./inputs(:,4);

    %find which samples are case 1 or case 2
    val_2 = find(ratio<0.85);
    val_1 = find(ratio>=0.85);

    %get case 1 and case 2 water type inputs
    x_case1 = inputs(val_1,[1:4,6:7]); 
    x_case2 = inputs(val_2,1:7);

    %pre allocate for input normalization
    x_case2_N = ones(size(x_case2));
    x_case1_N = ones(size(x_case1));

    %normalize input data for case 1 and case 2 waters
    for j = 1:7 
        x_case2_N(:,j) = (2/3)*((x_case2(:,j)-mu_2(j))/std_2(j));
    end
    for j = 1:6
        x_case1_N(:,j) = (2/3)*((x_case1(:,j)-mu_1(j))/std_1(j));
    end

    %Kd inversion for case 2
    [Kd_est_case2] = MLP_Kd(x_case2_N,w1_2,b1_2,w2_2,b2_2,wout_2,...
        bout_2,mu_2(8),std_2(8));

    %Kd inversion for case 1
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
% Computes the NN output from the inputs. Assumes 2 hidden layers, and
% utilizes a tanh activation function. The input are fed into the first
% layer as a matrix, then the second layer, then are collapsed to an
% output value. This function works for hidden, output, and input layers of
% any size, so long as the weight and bias functions are the right size and
% there are 2 hidden layers.
%
% Also note that the output of the neural net is assumed to be log10(Kd),
% and so it is raised to the power of 10 to ensure the correct output.
% Because Kd has a high dynamic range, the logarithm helps to ensure the 
% efficacy of the neural net. In addition, the output from the neural net
% is assumed to be normalized for efficacy, so denormalization constants 
% are added in to reverse this. 
%
% Inputs: x,w1,b1,w2,b2,wout,bout,muKd,stdKd
%       x (m x ni Double): The inputs to the neural net, with samples (m)
%       and number of inputs (ni). Note that for turbid and clear inputs,
%       the (ni) is different.
%		
%       w1 (nc1 x ni Double): Connection weights of the first hidden layer, 
%		which has (nc1) neurons and connects to the (ni) nuerons in the 
%		input layer.
%
%		b1 (nc1 x 1 Double): Neuron bias of the first hidden layer, which
%		has (nc1) neurons.
%
%		w2 (nc2 x nc1 Double): Connection weights of the second hidden
%		layer, which has (nc2) neurons. This connects to the (nc1) neurons
%		in the first hidden layer.
%
%		b2 (nc2 x 1 Double): Neuron bias of the second hidden layer, which
%		has (nc2) neurons.
%       
%       wout (1 x nc2 Double): Connection weights of the output layer, 
%       which connects to the (nc2) neurons of the second hidden layer and
%       returns only a single output.
%
%       bout (1x1 Double): Neuron bias of the output layer, which only has
%       1 neuron
%
%       muKd (1x1 Double): The mean of the output training values, for
%       denormalization of the output.
%
%       stdKd (1x1 Double): The std of the output training values, for
%       denormalization of the output.

% Outputs: Kd_est
%        Kd_est (m x 1 Double): The estimated Kd values for the (m) input
%        samples.
%
%Created: July 6, 2022
%Completed: July 12, 2022
%Updates: N/A
%
%Aster Taylor and Matthew Kehrli
%Ocean Optics Research Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%get the number of rows/samples
[rx,~]=size(x);

% Forwarding data through the NN:
% compute first hidden layer, using tanh activation function
a=1.715905*tanh(0.6666667*(x*w1'+(b1*ones(1,rx))'));
% compute second hidden layer, using tanh activation function
b=1.715905*tanh((2./3)*(a*w2'+(b2*ones(1,rx))'));
%compute final output 
y=b*wout'+bout*ones(rx,1);

% Denormalize, since NN is trained for normalized log-scale data
Kd_est=10.^(1.5*y*stdKd+muKd);
end
