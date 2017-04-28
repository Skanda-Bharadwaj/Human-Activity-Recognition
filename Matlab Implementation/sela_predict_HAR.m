%% Sequential Extreme Learning Algorithm - HAR :  Prediction 
%--------------------------------------------------------------------------
%  
%  HAR_test_6 : Loads testing data set : X.mat and y_test.mat
%
%  X_train.mat contains 60 different signals which are randomly sent into 
%  feedforwarding network.
%
%  y_train.mat contains labeled classes corresponding to the signals in
%  X_train.mat
%
%  Once all the signals are sent into the network the prediction accuracy
%  of the netwrok is calculated.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ======================= Predict Function ===============================

function [prediction_accuracy, confusion_matrix] = sela_predict_HAR( window_size, step_size,...
                                                                     a, b, X, y,...
                                                                     alpha, beta, gamma,...
                                                                     nnx_w1, nnx_w2, ...
                                                                     nny_w1, nny_w2, ...
                                                                     nnz_w1, nnz_w2)
                                                               
%% ====================== Useful Initializations ==========================

rand_num = randperm(size(X, 1)); 

predicted_matrix = zeros(size(X, 1), 3);
nnx_predict = zeros(1, size(nnx_w2, 1));
nny_predict = zeros(1, size(nnx_w2, 1));
nnz_predict = zeros(1, size(nnx_w2, 1));
prediction  = zeros(1, size(nnx_w2, 1));

%% ====================== Feed-Forward Netwrok ============================

for i = 1 : size(X, 1)
    
    signal = X{rand_num(i), 1};                           
    signal_length = length(signal);                                       
    last_integer = signal_length - window_size + (2*step_size);           

    nn_signal = [zeros(step_size, 3);...                                  
                 signal ;...                                                 
                 zeros((signal_length + step_size) - last_integer, 3)];   
    
    norm_nn_signal = zeros(size(nn_signal));
    for q = 1 : size(nn_signal, 2)
        norm_nn_signal(:, q) = mapminmax(nn_signal(:, q)');
    end
    
    signal_epoch = ceil(last_integer/step_size);
    yd = y{rand_num(i)};
    epoch = 0;
    nnx_output = zeros(signal_epoch, 6);
    nny_output = zeros(signal_epoch, 6);
    nnz_output = zeros(signal_epoch, 6);

    
    for j = 1 : step_size : last_integer 
        
            epoch = epoch + 1;
            
            nnx_x = norm_nn_signal(j : (j + window_size - 1), 1)';
            nny_x = norm_nn_signal(j : (j + window_size - 1), 2)';
            nnz_x = norm_nn_signal(j : (j + window_size - 1), 3)';
            
            m = size(nnx_x, 1);
            
            nnx_h1 = a * tanh(b * ([ones(m, 1) nnx_x] * nnx_w1'));
            nny_h1 = a * tanh(b * ([ones(m, 1) nny_x] * nny_w1'));
            nnz_h1 = a * tanh(b * ([ones(m, 1) nnz_x] * nnz_w1'));
            
            nnx_h2 = ([ones(m, 1) nnx_h1] * nnx_w2');
            nny_h2 = ([ones(m, 1) nny_h1] * nny_w2');
            nnz_h2 = ([ones(m, 1) nnz_h1] * nnz_w2');
            
            nnx_output(epoch, :) = nnx_h2;
            nny_output(epoch, :) = nny_h2;
            nnz_output(epoch, :) = nnz_h2;
            
            
    end

%% ======================= Class Prediction ===============================

    for k = 1 : size(nnx_output, 2)
        
        nnx_predict(k) = mean(nnx_output(:, k));
        nny_predict(k) = mean(nny_output(:, k));
        nnz_predict(k) = mean(nnz_output(:, k));
        
        prediction(k, :) = alpha * nnx_predict(k) + ...
                           beta  * nny_predict(k) + ...
                           gamma * nnz_predict(k);
    end

    prediction_max_value = max(prediction);
    
    for l = 1 : length(prediction)
        
        if (prediction(l) == prediction_max_value)
            predicted_class = l;
        end
        
    end
               
    predicted_matrix(i, :) = [yd,...
                              predicted_class,...
                              (yd == predicted_class)];
               
end

%% ======================= Prediction Accuracy ============================

prediction_accuracy = mean(predicted_matrix(:, 3))*100;
confusion_matrix = sela_HAR_confusion_matrix(predicted_matrix);

% ========================================================================
%% END
