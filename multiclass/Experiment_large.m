function  Experiment_large(dataset_name)
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
%% run experiments:
for i=1:size(ID_list,1),
    ID = ID_list(i,:);
    i

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

            % 9. BPAS
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = BPAs(Y,X, ID, options);
    nSV_bpas(i) = size_SV;
    err_bpas(i) = err_count;
    time_bpas(i) = run_time;
    mistakes_list_bpas(i,:) = mistakes;
    SVs_bpas(i,:) = SVs;
    TMs_bpas(i,:) = TMs;
    % 10. BOGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = BOGD(Y,X, ID, options);
    nSV_bogd(i) = size_SV;
    err_bogd(i) = err_count;
    time_bogd(i) = run_time;
    mistakes_list_bogd(i,:) = mistakes;
    SVs_bogd(i,:) = SVs;
    TMs_bogd(i,:) = TMs;

end
save('current');
%% print and plot results
figure
mean_mistakes_RBP = mean(mistakes_list_RBP);
plot(mistakes_idx, mean_mistakes_RBP,'r-p');
hold on
mean_mistakes_for = mean(mistakes_list_for);
plot(mistakes_idx, mean_mistakes_for,'r-x');
mean_mistakes_projectron = mean(mistakes_list_projectron);
plot(mistakes_idx, mean_mistakes_projectron,'r-o');
mean_mistakes_plus = mean(mistakes_list_plus);
plot(mistakes_idx, mean_mistakes_plus,'r-*');
mean_mistakes_bpas = mean(mistakes_list_bpas);
plot(mistakes_idx, mean_mistakes_bpas,'r-o');
mean_mistakes_bogd = mean(mistakes_list_bogd);
plot(mistakes_idx, mean_mistakes_bogd,'r-*');
mean_mistakes_FG = mean(mistakes_list_FG);
plot(mistakes_idx, mean_mistakes_FG,'b-d');
mean_mistakes_NY = mean(mistakes_list_NY);
plot(mistakes_idx, mean_mistakes_NY,'g-*');
legend('RBP','forgetron','projectron','projectron++','BPAS','BOGD','FOGD','NOGD');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
grid


figure
mean_TM_RBP = log(mean(TMs_RBP))/log(10);
plot(mistakes_idx, mean_TM_RBP,'r-p');
hold on
mean_TM_for = log(mean(TMs_for))/log(10);
plot(mistakes_idx, mean_TM_for,'r-x');
mean_TM_projectron = log(mean(TMs_projectron))/log(10);
plot(mistakes_idx, mean_TM_projectron,'r-o');
mean_TM_plus = log(mean(TMs_plus))/log(10);
plot(mistakes_idx, mean_TM_plus,'r-*');

mean_TM_bpas = log(mean(TMs_bpas))/log(10);
plot(mistakes_idx, mean_TM_bpas,'r-o');
mean_TM_bogd = log(mean(TMs_bogd))/log(10);
plot(mistakes_idx, mean_TM_bogd,'r-*');

mean_TM_FG = log(mean(TMs_FG))/log(10);
plot(mistakes_idx, mean_TM_FG,'b-d');
mean_TM_NY = log(mean(TMs_NY))/log(10);
plot(mistakes_idx, mean_TM_NY,'g-*');
legend('RBP','forgetron','projectron','projectron++','BPAS','BOGD','FOGD','NOGD');
xlabel('Number of samples');
ylabel('average time cost (log_{10} t)')
grid

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'RBP  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_RBP)*100, std(err_RBP)*100, mean(nSV_RBP), std(nSV_RBP), mean(time_RBP));
fprintf(1,'forgetron  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_for)*100, std(err_for)*100, mean(nSV_for), std(nSV_for), mean(time_for));
fprintf(1,'porjectron  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_projectron)*100, std(err_projectron)*100, mean(nSV_projectron), std(nSV_projectron), mean(time_projectron));
fprintf(1,'porjectron++  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_plus)*100, std(err_plus)*100, mean(nSV_plus), std(nSV_plus), mean(time_plus));
fprintf(1,'BPAS  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_bpas)*100, std(err_bpas)*100, mean(nSV_bpas), std(nSV_bpas), mean(time_bpas));
fprintf(1,'BOGD  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_bogd)*100, std(err_bogd)*100, mean(nSV_bogd), std(nSV_bogd), mean(time_bogd));
fprintf(1,'FOGD  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_FG)*100, std(err_FG)*100, mean(nSV_FG), std(nSV_FG), mean(time_FG));
fprintf(1,'NOGD  & %.3f $\\pm$ \t %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \\\\\n', mean(err_NY)*100, std(err_NY)*100, mean(nSV_NY), std(nSV_NY), mean(time_NY));
fprintf(1,'-------------------------------------------------------------------------------\n');

