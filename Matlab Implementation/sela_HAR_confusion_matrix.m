%% Sequential Extreme Learning Algorithm - HAR : Confusion Matrix 
%--------------------------------------------------------------------------
%  
%  Create the confusion matrix.
%  Target vs Output classes. Plot the matrix.
% 
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ======================= Confusion Matrix ===============================
function confusion_matrix = sela_HAR_confusion_matrix(predicted_matrix)

confusion_matrix = zeros(7, 7);
confusion_rows = zeros(6, 6);

for i = 1:6
    
    n1 = 0; n2 = 0; n3 = 0; 
    n4 = 0; n5 = 0; n6 = 0;
    
    for j = 1:size(predicted_matrix)
        
        if(predicted_matrix(j, 1) == i)
            
            switch predicted_matrix(j, 2)
                case 1,
                    n1 = n1+1;
                case 2,
                    n2 = n2+1;
                case 3,
                    n3 = n3+1;
                case 4,
                    n4 = n4+1;
                case 5,
                    n5 = n5+1;
                case 6,
                    n6 = n6+1;
                otherwise,
                    NULL;
            end
            
        end
                    
        confusion_rows(:, i) = [n1 n2 n3 n4 n5 n6];
        
    end
    
    confusion_matrix(7, i) = confusion_rows(i, i) .* 100 ./sum(confusion_rows(:, i));
    confusion_matrix(i, 7) = confusion_rows(i, i) .* 100 ./sum(confusion_rows(i, :));

end

confusion_matrix(7,7) = mean(predicted_matrix(:, 3))*100;
confusion_matrix(1:6, 1:6) = confusion_rows;

%% ==================== Plot Confusion Matrix =============================

% targets = zeros(6, 60);
% outputs = zeros(6, 60);
% 
% for ii = 1:6
%     for jj = 1:size(predicted_matrix, 1)
%         if (predicted_matrix(jj, 1) == ii)
%             targets(ii, jj) = 1;
%         else
%             targets(ii, jj) = 0;
%         end
%     end
% end
% 
% for iii = 1:6
%     for jjj = 1:size(predicted_matrix, 1)
%         if (predicted_matrix(jjj, 2) == iii)
%             outputs(iii, jjj) = 1;
%         else
%             outputs(iii, jjj) = 0;
%         end
%     end
% end

% plotconfusion(targets, outputs, 'HAR TRAINING');
set(findall(0,'FontName','Helvetica','FontSize',10),...
    'FontName','Times New Roman','FontSize',12);
% =========================================================================
%% End