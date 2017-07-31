function  Experiment_D(dataset_name)
% Experiment: the main procedure evaluating all the algorithm on the same dataset
%--------------------------------------------------------------------------
% Input:
%      dataset_name: the name of the dataset file
% Output:
%      a figure of online mistake rates for all the algorithms
%      a figure of online SV size for all the algorithms
%      a figure of online time consumption for all the algorithms
%      a table of the final mistake rates, SV size, time consumption for all the algorithms
%--------------------------------------------------------------------------

%% load dataset
load(sprintf('data/%s',dataset_name));
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);
%% options
options.C=10;
options.rho=0.2;% rho \in [0,1]
options.eta    =  0.5;%learning rate
options.Budget=100;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;
k=5:5:30;
%% run experiments:
for j=1:size(k,2)
j
options.rou_f=k(j);
for i=1:10,
    ID = ID_list(i,:);
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = FouGD(Y, X, ID, options);
    nSV_FG(i) = size_SV;
    err_FG(i) = err_count;
    time_FG(i) = run_time;
    mistakes_list_FG(i,:) = mistakes;
    SVs_FG(i,:) = SVs;
    TMs_FG(i,:) = TMs;
end
time(j)=mean(time_FG);
err(j)=mean(err_FG);
end
figure
plot(k,time);
figure
plot(k,err);