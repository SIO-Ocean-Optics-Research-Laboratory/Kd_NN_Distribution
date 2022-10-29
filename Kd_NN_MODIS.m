function [Kd]=Kd_NN_MODIS(sza,lambda,Rrs,Kd_LUT)
%Implements the neural network (NN) algorithm to calculate the diffuse 
%attenuation coefficient of downwelling planar irradiance (Kd) at one 
%preselected output light wavelength (lambda) using input remote-sensing 
%reflectance (Rrs) at MODIS wavelengths and solar zenith angle (sza)
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
%%original NN algorithm presented in Jamet et al. (2012)
%
%Required function inputs: R_rs, sza, lambda, Kd_LUT
%   Rrs [1x5 Double]: Values of spectral remote-sensing reflectance [sr^-1]
%   at MODIS light wavelengths: 443, 488, 531, 547, 667 [nm].
%
%   sza [1x1 Double]: Solar zenith angle [deg] associated with input Rrs
%   values.
%
%   lambda [1x1 Double]: Output light wavelength [nm] at which the desired
%   value of Kd is estimated for a given input. Lambda serves as an input
%   parameter for the Kd_NN function and is defined by user. Note: light
%   wavelength is in vacuum
%
%   Kd_LUT [1x1 Structure]: Structure containing three required look-up
%   tables (LUTs); can be loaded via load('Kd_NN_LUT.mat')
%
%       Kd_LUT.weights_1: LUT with weights and biases from NN for clear
%       waters (where Rrs(488)/Rrs(547) >= 0.85)
%
%       Kd_LUT.weights_2: LUT with weights and biases from NN for turbid
%       waters (where Rrs(488)/Rrs(547) < 0.85)
%
%       Kd_LUT.train_switch: LUT with means and standard deviations of 
%       40,000 inputs and outputs used to train the NN
%
%Outputs: Kd
%   Kd [1x1 Double]: The estimated value of the average diffuse attenuation
%   coefficient of downwelling planar irradiance [m^-1] between the sea
%   surface and first attenuation depth at the output light wavelength
%   (lambda) for input spectral Rrs and sza.
%
%Version History: 
%2018-04-04: Original implementation in C written by David Dessailly
%2020-03-23: Original Matlab version, D. Jorge 
%2022-09-01: Revised Matlab version, M. Kehrli
%DARIUSZ: WHEN WE ARE FINISHED, ADD DATE AND Final Revised Matab version: M. Kehrli, R. A. Reynolds, and D. Stramski
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check function arguments and existence of LUTs
    arguments
        sza (1,1) double 
        lambda (1,1) double
        Rrs (1,5) double
        Kd_LUT (1,1) struct
    end
    
    %refractive index of seawater
    nw = 1.34;
    %calculation of muw [dim], the cosine of the angle of refraction of the
    %solar beam just beneath the sea surface
    muw = cosd(asind(sind(sza)/nw));
    
    %combine inputs into a single array for NN
    inputs = [Rrs,lambda,muw];
    
    %mean and standard deviation of input and output parameters from LUT
    %for each NN; determined from training dataset of 40,000 inputs and
    %outputs to normalize NN
    train_switch = Kd_LUT.train_switch;
    mu_switch = train_switch.('MEAN')';
    std_switch = train_switch.('STD')';
    
    %calculate the blue-green band ratio of reflectance to determine water
    %type of the input Rrs spectrum
    ratio = inputs(2)/inputs(4);
    
    %build NN for clear waters
    if ratio >= 0.85 
        %read in NN weights and biases for clear waters
        weights_1 = Kd_LUT.weights_1;
        
        %number of input neurons in the NN
        ne = 6;
        %number of neurons on the first hidden layer in the NN
        nc1 = 8;
        %number of neurons on the second hidden layer in the NN
        nc2 = 6;
        %number of neurons on the output layer in the NN
        ns = 1;
        
        %set biases from LUT bias matricies 
        b1 = weights_1.('b1'); b1(isnan(b1)) = [];
        b2 = weights_1.('b2'); b2(isnan(b2)) = [];
        bout = weights_1.('bout'); bout(isnan(bout)) = [];
        %set weights from LUT weight matricies 
        w1 = weights_1.('w1'); w1(isnan(w1)) = []; w1 = reshape(w1,nc1,ne);
        w2 = weights_1.('w2');w2(isnan(w2)) = []; w2 = reshape(w2,nc2,nc1);
        wout = weights_1.('wout'); wout(isnan(wout)) = []; 
        wout = reshape(wout,ns,nc2);
        
        %mean and stadard deviation of input and output parameters from
        %training dataset; remove Rrs(667) data for clear waters
        mu = mu_switch([2:5,7:9]);
        std = std_switch([2:5,7:9]);
        
        %set NN input for clear waters
        x = inputs([1:4,6:7]);
        
        %if any Rrs input to the NN is negative, set the function output to
        %NaN, send a warning message, and return
        if any(x(1:4)<0)
            Kd = nan;
            warning(['At least one input of Rrs is negative.'...
                'Output Kd set to nan.'])
            return
        end
        
        %preallocate for input normalization
        x_N = ones(size(x));
        
        %normalize input data
        for j = 1:6
            x_N(:,j) = (2/3)*((x(:,j)-mu(j))/std(j));
        end
    
    %Build NN for turbid waters
    elseif ratio < 0.85
        %read in NN weights and biases for turbid waters
        weights_2 = Kd_LUT.weights_2;

        %number of input neurons in the NN
        ne = 7;
        %number of neurons on the first hidden layer in the NN
        nc1 = 9;
        %number of neurons on the second hidden layer in the NN
        nc2 = 6;
        %number of neurons on the output layer in the NN
        ns = 1;
        
        %set biases from LUT bias matricies 
        b1 = weights_2.('b1'); b1(isnan(b1)) = [];
        b2 = weights_2.('b2'); b2(isnan(b2)) = [];
        bout = weights_2.('bout'); bout(isnan(bout)) = [];
        %set weights from LUT weight matricies 
        w1 = weights_2.('w1'); w1(isnan(w1)) = []; w1=reshape(w1,nc1,ne);
        w2 = weights_2.('w2'); w2(isnan(w2)) = []; w2=reshape(w2,nc2,nc1);
        wout = weights_2.('wout'); wout(isnan(wout)) = [];
        wout=reshape(wout,ns,nc2);
        
        %mean and stadard deviation of input and output parameters from
        %training dataset
        mu = mu_switch(2:9);
        std = std_switch(2:9);
        
        %set NN input for turbid waters
        x = inputs;
        
        %if any Rrs input to the NN is negative, set the function output to
        %NaN, send a warning message, and return
        if any(x(1:5)<0)
            Kd = nan;
            warning(['At least one input of Rrs is negative.'...
                'Output Kd set to nan.'])
            return
        end
        
        %preallocate for input normalization
        x_N = ones(size(x));
        
        %normalize input data
        for j = 1:7 
            x_N(:,j) = (2/3)*((x(:,j)-mu(j))/std(j));
        end
    end

%Kd inversion
[Kd] = MLP_Kd(x_N,w1,b1,w2,b2,wout,bout,mu(end),std(end));
    
end
%end of main code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Additional subfunctions that are called
function [Kd]=MLP_Kd(x,w1,b1,w2,b2,wout,bout,muKd,stdKd)
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
%   x [1 x ni Double]: The inputs to the NN and the number of input
%   neurons, ni. Note that six input neurons are used for clear waters and
%   seven input neurons are used for turbid waters
%		
%   w1 [nc1 x ni Double]: Connection weights of the first hidden
%	layer, which has (nc1) neurons and connects to the (ni) neurons
%	in the input layer
%
%	b1 [nc1 x 1 Double]: Neuron bias of the first hidden layer, 
%	which has (nc1) neurons
%
%	w2 [nc2 x nc1 Double]: Connection weights of the second hidden
%	layer, which has (nc2) neurons. This connects to the (nc1) 
%   neurons in the first hidden layer
%
%	b2 [nc2 x 1 Double]: Neuron bias of the second hidden layer,
%   which has (nc2) neurons
%       
%   wout [1 x nc2 Double]: Connection weights of the output layer, 
%   which connects to the (nc2) neurons of the second hidden layer
%	and returns only a single output
%
%   bout [1x1 Double]: Neuron bias of the output layer, which only 
%   has 1 neuron
%
%   muKd [1x1 Double]: The mean output of Kd values from NN training, for
%   denormalization of the output
%
%   stdKd [1x1 Double]: The std output of Kd values from NN training, for
%   denormalization of the output
%
%Outputs: Kd
%   Kd [1 x 1 Double]: The estimated Kd value obtained from the NN
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
Kd=10.^(1.5*y*stdKd+muKd);
end
