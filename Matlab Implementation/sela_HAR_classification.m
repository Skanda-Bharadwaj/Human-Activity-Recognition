%% Sequential Extreme Learning Algorithm : Human Activity Recognition (HAR)
%--------------------------------------------------------------------------
%  
%  The project aims at classifying different human activities such as  
%  walking, sitting, standing, lying etc. The network is designed for
%  a 6-class problem
%
%  Sequential Extreme Learning Algorithm (SELA) is used to train the 
%  neural network. A 3D neural network is designed to learn each of the  
%  axis independently (x, y & z) and finally the result is combined to 
%  increase the prediction accuracy.
%
%  The input layer and the output layer of the network remains same for
%  all the axis. However, hidden layers can be designed individually 
%  corresponding to each of the axis. 
%
%  labels for classes of recognition are as follows :
%  1. Walking
%  2. Walking upstairs
%  3. Walking downstairs
%  4. Sitting
%  5. Standing
%  6. Lying 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ========================== Initialize ==================================

clear; close all; clc;

%% ======================= Load Training Database =========================

load ('HAR_train_6.mat');
load ('HAR_test_6.mat');

%% ======================= Neural Network Model ===========================

window_size  = 20;                                                        
step_size    = 5;

input_layer  = window_size;
nnx_hidden_layer = 120; %120
nny_hidden_layer = 120; %120
nnz_hidden_layer = 120; %120
output_layer = 6;

alpha = 0.4;  %3
beta  = 0.41; %2.6
gamma = 0.19; %1.99
% ==================== Activation Field Parameters ========================

a = 1.7159;
b = 2/3;

% ================== Initializing Layer-1 Weights =========================

nnx_w1 = sela_random_weights(input_layer, nnx_hidden_layer);
nny_w1 = sela_random_weights(input_layer, nny_hidden_layer);
nnz_w1 = sela_random_weights(input_layer, nnz_hidden_layer);

nnx_w2 = zeros(output_layer, nnx_hidden_layer+1);
nny_w2 = zeros(output_layer, nny_hidden_layer+1);
nnz_w2 = zeros(output_layer, nnz_hidden_layer+1);

% ================== Initializing lambda and P ============================

lambda = 10^-6;

nnx_p = 1/lambda * eye(nnx_hidden_layer+1);
nny_p = 1/lambda * eye(nny_hidden_layer+1);
nnz_p = 1/lambda * eye(nnz_hidden_layer+1);

%% ============= Sequential Extreme Learning Algorithm ====================
random_num = randperm(size(X_train, 1));
for i = 1 : size(X_train, 1)
    
    signal = X_train{random_num(i), 1};
    signal_length = length(signal);
    last_integer = signal_length - window_size + (2*step_size);
    nn_signal = [zeros(step_size, 3);...                                  
                 signal;...                                                 
                 zeros((signal_length + step_size) - last_integer, 3)];
             
    norm_nn_signal = zeros(size(nn_signal));
    for kk = 1 : size(nn_signal, 2)
        norm_nn_signal(:, kk) = mapminmax(nn_signal(:, kk)');
    end
     
    y  = y_train(random_num(i));                                          %  Network conventions :
    yd = ((1 : output_layer) == y{1, 1})';                                %  nnx - neural net for x - axis
                                                                          %  nny - neural net for y - axis
    for j = 1 : step_size : last_integer                                  %  nnz - neural net for z - axis
        
        nnx_x = norm_nn_signal(j : (j + window_size - 1), 1)';            %  x - Chuck of signal (part of the signal with length equal to window size)
        nny_x = norm_nn_signal(j : (j + window_size - 1), 2)';
        nnz_x = norm_nn_signal(j : (j + window_size - 1), 3)';
        
        nnx_x1 = [1, nnx_x]';                                             %  x1 - Output of layer 1 with bias node
        nny_x1 = [1, nny_x]';
        nnz_x1 = [1, nnz_x]';
        
        nnx_v1 = nnx_w1 * nnx_x1;                                         %  v1 - Input to layer 2
        nny_v1 = nny_w1 * nny_x1;
        nnz_v1 = nnz_w1 * nnz_x1;
        
        nnx_y1 = a * tanh(b * nnx_v1);                                    %  y1 - Input to layer 2 
        nny_y1 = a * tanh(b * nny_v1);
        nnz_y1 = a * tanh(b * nnz_v1);
    
        nnx_x2 = [1; nnx_y1];                                             %  x2 - Output of layer 2 with bias node
        nny_x2 = [1; nny_y1];
        nnz_x2 = [1; nnz_y1];
    
        nnx_p  = nnx_p - (nnx_p * (nnx_x2 * nnx_x2') * nnx_p) ./ (1 + (nnx_x2' * nnx_p * nnx_x2));      
        nny_p  = nny_p - (nny_p * (nny_x2 * nny_x2') * nny_p) ./ (1 + (nny_x2' * nny_p * nny_x2));
        nnz_p  = nnz_p - (nnz_p * (nnz_x2 * nnz_x2') * nnz_p) ./ (1 + (nnz_x2' * nnz_p * nnz_x2));
        
        nnx_w2 = nnx_w2 + ((yd - (nnx_w2 * nnx_x2)) * (nnx_x2' * nnx_p));
        nny_w2 = nny_w2 + ((yd - (nny_w2 * nny_x2)) * (nny_x2' * nny_p));
        nnz_w2 = nnz_w2 + ((yd - (nnz_w2 * nnz_x2)) * (nnz_x2' * nnz_p));
        
    end
    
end

%% =================== Function call for Testing ==========================
figure(1)
[training_prediction_accuracy, training_confusion_matrix] = sela_predict_HAR( window_size, step_size,... 
                                                            a, b, X_train, y_train,...
                                                            alpha, beta, gamma,...
                                                            nnx_w1, nnx_w2,...
                                                            nny_w1, nny_w2,...
                                                            nnz_w1, nnz_w2);
figure(2)                                          
[testing__prediction_accuracy, testing_confusion_matrix] = sela_predict_HAR( window_size, step_size,... 
                                                            a, b, X_test, y_test,...
                                                            alpha, beta, gamma,...
                                                            nnx_w1, nnx_w2,...
                                                            nny_w1, nny_w2,...
                                                            nnz_w1, nnz_w2);

% =========================================================================
%%  END
training_confusion_matrix
testing_confusion_matrix