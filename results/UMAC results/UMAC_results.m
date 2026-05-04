clear all;close all; clc;
load('results_all.mat');

%% cvxEDA results
results = results_cvx;
method_names = fieldnames(results);

for m = 1:length(method_names)
    method = method_names{m};

    fprintf('\n%s\n', method);
    fprintf('MAE:  %.3f ± %.3f\n', mean(results.(method).MAE),  std(results.(method).MAE));
    fprintf('RMSE: %.3f ± %.3f\n', mean(results.(method).RMSE), std(results.(method).RMSE));
    fprintf('PCC:  %.3f ± %.3f\n', mean(results.(method).PCC),  std(results.(method).PCC));
    fprintf('SNR:  %.3f ± %.3f\n', mean(results.(method).SNR),  std(results.(method).SNR));
end

%% ospEDA results
results = results_cvx;
method_names = fieldnames(results);

for m = 1:length(method_names)
    method = method_names{m};

    fprintf('\n%s\n', method);
    fprintf('MAE:  %.3f ± %.3f\n', mean(results.(method).MAE),  std(results.(method).MAE));
    fprintf('RMSE: %.3f ± %.3f\n', mean(results.(method).RMSE), std(results.(method).RMSE));
    fprintf('PCC:  %.3f ± %.3f\n', mean(results.(method).PCC),  std(results.(method).PCC));
    fprintf('SNR:  %.3f ± %.3f\n', mean(results.(method).SNR),  std(results.(method).SNR));
end