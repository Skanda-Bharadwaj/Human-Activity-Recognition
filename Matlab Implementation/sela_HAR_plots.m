load ('HAR_train_6.mat');

walking = X_train{1, 1};
walking_upstairs = X_train{118, 1};
walking_downstairs = X_train{291, 1};
sitting = X_train{467, 1};
standing = X_train{577, 1};
lying_down = X_train{687, 1};

figure(1)
subplot(231),plot(walking(:, 1),'r-','LineWidth',2); hold on
subplot(231),plot(walking(:, 2),'g-','LineWidth',2); hold on
subplot(231),plot(walking(:, 3),'b-','LineWidth',2);
axis([0 600 -0.8 1.7]);
xlabel('Sample #'); ylabel('Amplitude'); title('Walking');

subplot(232),plot(walking_upstairs(:, 1),'r-','LineWidth',2); hold on
subplot(232),plot(walking_upstairs(:, 2),'g-','LineWidth',2); hold on
subplot(232),plot(walking_upstairs(:, 3),'b-','LineWidth',2);
axis([0 650 -0.85 1.55]);
xlabel('Sample #'); ylabel('Amplitude'); title('Walking-upstairs');

subplot(233),plot(walking_downstairs(:, 1),'r-','LineWidth',2); hold on
subplot(233),plot(walking_downstairs(:, 2),'g-','LineWidth',2); hold on
subplot(233),plot(walking_downstairs(:, 3),'b-','LineWidth',2);
axis([0 700 -0.95 2]);
xlabel('Sample #'); ylabel('Amplitude'); title('Walking-downstairs');

subplot(234),plot(sitting(:, 1),'r-','LineWidth',2); hold on
subplot(234),plot(sitting(:, 2),'g-','LineWidth',2); hold on
subplot(234),plot(sitting(:, 3),'b-','LineWidth',2);
axis([0 810 0.12 1.08]);
xlabel('Sample #'); ylabel('Amplitude'); title('Sitting');

subplot(235),plot(standing(:, 1),'r-','LineWidth',2); hold on
subplot(235),plot(standing(:, 2),'g-','LineWidth',2); hold on
subplot(235),plot(standing(:, 3),'b-','LineWidth',2);
axis([0 1000 -0.175 1.06]);
xlabel('Sample #'); ylabel('Amplitude'); title('Standing');

subplot(236),plot(lying_down(:, 1),'r-','LineWidth',2); hold on
subplot(236),plot(lying_down(:, 2),'g-','LineWidth',2); hold on
subplot(236),plot(lying_down(:, 3),'b-','LineWidth',2);
axis([0 900 0.13 0.85]);
xlabel('Sample #'); ylabel('Amplitude'); title('Lying down');

set(findall(0,'FontName','Helvetica','FontSize',10),...
    'FontName','Times New Roman','FontSize',12);