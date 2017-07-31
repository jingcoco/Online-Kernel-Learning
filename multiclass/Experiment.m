function  Experiment(dataset_name)
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
options.eta_fou=0.005;
options.rou_f=4;
options.eta    =  0.5;%learning rate
options.Budget=200;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;
%% run experiments:  
for i=1:2,
    i
    ID = ID_list(i,:);
        % 1. Max perceptron
    [err_count, run_time, mistakes, mistakes_idx, SVs,size_SV, TMs] = perceptron(Y,X, ID,options);

    nSV_MA(i) = size_SV;
    err_MA(i) = err_count;
    time_MA(i) = run_time;
    mistakes_list_MA(i,:) = mistakes;
    SVs_MA(i,:) = SVs;
    TMs_MA(i,:) = TMs;

    %. OGD
  
    [err_count, run_time, mistakes, mistakes_idx, SVs,size_SV, TMs] = ogd(Y,X, ID,options);   
    nSV_OGD(i) = size_SV;
    err_OGD(i) = err_count;
    time_OGD(i) = run_time;
    mistakes_list_OGD(i,:) = mistakes;
    SVs_OGD(i,:) = SVs;
    TMs_OGD(i,:) = TMs;
    
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = FouGD(Y, X, ID, options);
    nSV_FG(i) = size_SV;
    err_FG(i) = err_count;
    time_FG(i) = run_time;
    mistakes_list_FG(i,:) = mistakes;
    SVs_FG(i,:) = SVs;
    TMs_FG(i,:) = TMs;

    % 5. NyGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y,X, ID, options);
    nSV_NY(i) = size_SV;
    err_NY(i) = err_count;
    time_NY(i) = run_time;
    mistakes_list_NY(i,:) = mistakes;
    SVs_NY(i,:) = SVs;
    TMs_NY(i,:) = TMs;
    % 7. RBP
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = RBP(Y,X, ID, options);
    nSV_RBP(i) = size_SV;
    err_RBP(i) = err_count;
    time_RBP(i) = run_time;
    mistakes_list_RBP(i,:) = mistakes;
    SVs_RBP(i,:) = SVs;
    TMs_RBP(i,:) = TMs;
    % 8. forgetron
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = forgetron(Y,X, ID, options);
    nSV_for(i) = size_SV;
    err_for(i) = err_count;
    time_for(i) = run_time;
    mistakes_list_for(i,:) = mistakes;
    SVs_for(i,:) = SVs;
    TMs_for(i,:) = TMs;
        % 9. projectron
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = projectron(Y,X, ID, options);
    nSV_projectron(i) = size_SV;
    err_projectron(i) = err_count;
    time_projectron(i) = run_time;
    mistakes_list_projectron(i,:) = mistakes;
    SVs_projectron(i,:) = SVs;
    TMs_projectron(i,:) = TMs;
    % 10. projectron++
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = proplus(Y,X, ID, options);
    nSV_plus(i) = size_SV;
    err_plus(i) = err_count;
    time_plus(i) = run_time;
    mistakes_list_plus(i,:) = mistakes;
    SVs_plus(i,:) = SVs;
    TMs_plus(i,:) = TMs;
    
        % 11. BPAs
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = BPAs(Y,X, ID, options);
    nSV_BPAS(i) = size_SV;
    err_BPAS(i) = err_count;
    time_BPAS(i) = run_time;
    mistakes_list_BPAS(i,:) = mistakes;
    SVs_BPAS(i,:) = SVs;
    TMs_BPAS(i,:) = TMs;
        % 11. BOGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = BOGD(Y,X, ID, options);
    nSV_BOGD(i) = size_SV;
    err_BOGD(i) = err_count;
    time_BOGD(i) = run_time;
    mistakes_list_BOGD(i,:) = mistakes;
    SVs_BOGD(i,:) = SVs;
    TMs_BOGD(i,:) = TMs;

end
%% print and plot results
figure
mean_mistakes_MA = mean(mistakes_list_MA);
plot(mistakes_idx, mean_mistakes_MA,'b.-');
hold on
mean_mistakes_OGD = mean(mistakes_list_OGD);
plot(mistakes_idx, mean_mistakes_OGD,'b-v');
mean_mistakes_RBP = mean(mistakes_list_RBP);
plot(mistakes_idx, mean_mistakes_RBP,'r-p');
mean_mistakes_for = mean(mistakes_list_for);
plot(mistakes_idx, mean_mistakes_for,'r-x');
mean_mistakes_projectron = mean(mistakes_list_projectron);
plot(mistakes_idx, mean_mistakes_projectron,'r-o');
mean_mistakes_plus = mean(mistakes_list_plus);
plot(mistakes_idx, mean_mistakes_plus,'r-*');
mean_mistakes_BPAS = mean(mistakes_list_BPAS);
plot(mistakes_idx, mean_mistakes_BPAS,'r-s');
mean_mistakes_BOGD = mean(mistakes_list_BOGD);
plot(mistakes_idx, mean_mistakes_BOGD,'r-^');
mean_mistakes_FG = mean(mistakes_list_FG);
plot(mistakes_idx, mean_mistakes_FG,'m-d');
mean_mistakes_NY = mean(mistakes_list_NY);
plot(mistakes_idx, mean_mistakes_NY,'m-<');
legend('Perceptron','OGD','RBP','forgetron','projectron','projectron++','BPAS','BOGD','FOGD','NOGD');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
grid

figure
mean_TM_MA = log(mean(TMs_MA))/log(10);
plot(mistakes_idx, mean_TM_MA,'b.-');
hold on
mean_TM_OGD = log(mean(TMs_OGD))/log(10);
plot(mistakes_idx, mean_TM_OGD,'b-v');
mean_TM_RBP = log(mean(TMs_RBP))/log(10);
plot(mistakes_idx, mean_TM_RBP,'r-p');
mean_TM_for = log(mean(TMs_for))/log(10);
plot(mistakes_idx, mean_TM_for,'r-x');
mean_TM_projectron = log(mean(TMs_projectron))/log(10);
plot(mistakes_idx, mean_TM_projectron,'r-o');
mean_TM_plus = log(mean(TMs_plus))/log(10);
plot(mistakes_idx, mean_TM_plus,'r-*');
mean_TM_BPAS = log(mean(TMs_BPAS))/log(10);
plot(mistakes_idx, mean_TM_BPAS,'r-s');
mean_TM_BOGD = log(mean(TMs_BOGD))/log(10);
plot(mistakes_idx, mean_TM_BOGD,'r-^');
mean_TM_FG = log(mean(TMs_FG))/log(10);
plot(mistakes_idx, mean_TM_FG,'m-d');
mean_TM_NY = log(mean(TMs_NY))/log(10);
plot(mistakes_idx, mean_TM_NY,'m-<');
legend('Perceptron','OGD','RBP','forgetron','projectron','projectron++','BPAS','BOGD','FOGD','NOGD');
xlabel('Number of samples');
ylabel('average time cost (log_{10} t)')
grid
fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'(mistakes rate, size of support vectors, cpu running time)\n');
fprintf(1,'Perceptron  & %.3f $\\pm$ \t %.3f & \t& %.3f \\\\\n', mean(err_MA)*100, std(err_MA)*100,mean(time_MA));
fprintf(1,'OGD  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_OGD)*100, std(err_OGD)*100,mean(time_OGD));
fprintf(1,'RBP  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_RBP)*100, std(err_RBP)*100, mean(time_RBP));
fprintf(1,'forgetron  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_for)*100, std(err_for)*100,mean(time_for));
fprintf(1,'porjectron  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_projectron)*100, std(nSV_projectron), mean(time_projectron));
fprintf(1,'porjectron++  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_plus)*100, std(err_plus)*100,mean(time_plus));
fprintf(1,'BPAS  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_BPAS)*100, std(err_BPAS)*100, mean(time_BPAS));
fprintf(1,'BOGD  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_BOGD)*100, std(err_BOGD)*100,  mean(time_BOGD));
fprintf(1,'FOGD  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_FG)*100, std(err_FG)*100, mean(time_FG));
fprintf(1,'NOGD  & %.3f $\\pm$ \t %.3f \t& %.3f \\\\\n', mean(err_NY)*100, std(err_NY)*100, mean(time_NY));
fprintf(1,'-------------------------------------------------------------------------------\n');

