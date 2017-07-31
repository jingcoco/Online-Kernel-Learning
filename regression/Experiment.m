function  Experiment(dataset_name)
% Experiment: the main procedure evaluating all the algorithm on the same dataset
%--------------------------------------------------------------------------
% Input:
%      dataset_name: the name of the dataset file
% Output:
%      a figure of online mistake rates for all the algorithms
%      a figure of online time consumption for all the algorithms
%      a table of the final mistake rates, time consumption for all the algorithms
%--------------------------------------------------------------------------

%% load dataset
load(sprintf('data/%s',dataset_name));
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);
%% options
options.C=10;
options.rho=0.2;% rho \in [0,1]
options.Budget=30;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.t_tick=round(n/20); 
options.gamma  =  10;
options.epsl=0.1;
options.eta=0.1;%%%%%%%%%%%%
options.nu=0.4;
options.Epsilon_eta=10^(-2)*3;%%%%%%%%%%%
options.eta_perceptron=0.1;
options.Fou_eta=10^(-4)*6;%%%%%%%%%%%
options.lambda=0.03;%%%%%%%%%%%%%%
options.k_f=15;
%% run experiments:
for i=1:20,
    i
    ID = ID_list(i,:);
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =perceptron(Y, X,ID, options);

    err_per(i) = err_count;
    time_per(i) = run_time;


    TMs_per(i,:) = TMs;
    loss_avg_per(i)=loss_avg;
    loss_v_per(i,:)=loss_v;
    % 1. OGD

    err_OGD(i) = err_count;
    time_OGD(i) = run_time;


    TMs_OGD(i,:) = TMs;
    loss_avg_OGD(i)=loss_avg;
    loss_v_OGD(i,:)=loss_v;
    
    % 2. epsilon
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =Epsilon(Y, X,ID, options);

    err_eps(i) = err_count;
    time_eps(i) = run_time;


    TMs_eps(i,:) = TMs;
    loss_avg_eps(i)=loss_avg;
    loss_v_eps(i,:)=loss_v;

            % 2.RBP
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =RBP(Y, X,ID, options);

    err_RBP(i) = err_count;
    time_RBP(i) = run_time;


    TMs_RBP(i,:) = TMs;
    loss_avg_RBP(i)=loss_avg;
    loss_v_RBP(i,:)=loss_v;
    %forgetron
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =forgetron(Y, X,ID, options);

    err_fog(i) = err_count;
    time_fog(i) = run_time;


    TMs_fog(i,:) = TMs;
    loss_avg_fog(i)=loss_avg;
    loss_v_fog(i,:)=loss_v;
      %projectron  
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =projectron(Y, X,ID, options);

    err_prj(i) = err_count;
    time_prj(i) = run_time;


    TMs_prj(i,:) = TMs;
    loss_avg_prj(i)=loss_avg;
    loss_v_prj(i,:)=loss_v;
          %FouGD 
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =FouGD(Y, X,ID, options);

    err_fou(i) = err_count;
    time_fou(i) = run_time;


    TMs_fou(i,:) = TMs;
    loss_avg_fou(i)=loss_avg;
    loss_v_fou(i,:)=loss_v;
              %NyGD 
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =NysGD(Y, X,ID, options);

    err_ny(i) = err_count;
    time_ny(i) = run_time;


    TMs_ny(i,:) = TMs;
    loss_avg_ny(i)=loss_avg;
    loss_v_ny(i,:)=loss_v;
    %BOGD
    [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =BOGD(Y, X,ID, options);
    err_bg(i) = err_count;
    time_bg(i) = run_time;


    TMs_bg(i,:) = TMs;
    loss_avg_bg(i)=loss_avg;
    loss_v_bg(i,:)=loss_v;    
end
%% print and plot results
figure
mean_TM_OGD = log(mean(TMs_OGD))/log(10);
plot(mistakes_idx, mean_TM_OGD,'b-v');
hold on
mean_TM_per = log(mean(TMs_per))/log(10);
plot(mistakes_idx, mean_TM_per,'b->');
 mean_TM_eps = log(mean(TMs_eps))/log(10);
 plot(mistakes_idx, mean_TM_eps,'b-+');
mean_TM_RBP = log(mean(TMs_RBP))/log(10);
plot(mistakes_idx, mean_TM_RBP,'r-p');
mean_TM_fog = log(mean(TMs_fog))/log(10);
plot(mistakes_idx, mean_TM_fog,'r-x');
mean_TM_prj = log(mean(TMs_prj))/log(10);
plot(mistakes_idx, mean_TM_prj,'r-o');
mean_TM_bg = log(mean(TMs_bg))/log(10);
plot(mistakes_idx, mean_TM_bg,'r-^');
mean_TM_fou = log(mean(TMs_fou))/log(10);
plot(mistakes_idx, mean_TM_fou,'m-d');
mean_TM_ny = log(mean(TMs_ny))/log(10);
plot(mistakes_idx, mean_TM_ny,'m-<');
legend('OGD','Perceptron','Norma','RBP','forgetron','projectron','BOGD','FOGD','NOGD');
xlabel('Number of samples');
ylabel('average time cost (log_{10} t)')
grid

figure
mean_loss_OGD=mean(loss_v_OGD);
plot(mistakes_idx, mean_loss_OGD,'b-v');
hold on
mean_loss_per=mean(loss_v_per);
plot(mistakes_idx, mean_loss_per,'b->');
mean_loss_eps=mean(loss_v_eps);
plot(mistakes_idx, mean_loss_eps,'b-+');
mean_loss_RBP=mean(loss_v_RBP);
plot(mistakes_idx, mean_loss_RBP,'r-p');
mean_loss_fog=mean(loss_v_fog);
plot(mistakes_idx, mean_loss_fog,'r-x');
mean_loss_prj=mean(loss_v_prj);
plot(mistakes_idx, mean_loss_prj,'r-o');
mean_loss_bg=mean(loss_v_bg);
plot(mistakes_idx, mean_loss_bg,'r-^');
mean_loss_fou=mean(loss_v_fou);
plot(mistakes_idx, mean_loss_fou,'m-d');
mean_loss_ny=mean(loss_v_ny);
plot(mistakes_idx, mean_loss_ny,'m-<');
legend('OGD','Perceptron','Norma','RBP','forgetron','projectron','BOGD','FOGD','NOGD');
xlabel('Number of samples');
ylabel('Average loss');
grid

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'OGD   & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n', mean(loss_avg_OGD),std(loss_avg_OGD),mean(time_OGD));
fprintf(1,'Perceptron & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n', mean(loss_avg_per),std(loss_avg_per),mean(time_per));
fprintf(1,'Norma & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_eps),std(loss_avg_eps),mean(time_eps));
fprintf(1,'RBP  & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_RBP),std(loss_avg_RBP), mean(time_RBP));
fprintf(1,'fogetron  & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_fog),std(loss_avg_fog), mean(time_fog));
fprintf(1,'projetron   & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_prj),std(loss_avg_prj),mean(time_prj));
fprintf(1,'BOGD  & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_bg),std(loss_avg_bg),mean(time_bg));
fprintf(1,'FOGD   & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_fou),std(loss_avg_fou),mean(time_fou));
fprintf(1,'NOGD  & %.5f $\\pm$ \t %.5f \t& %.5f \\\\\n',  mean(loss_avg_ny),std(loss_avg_ny), mean(time_ny));
fprintf(1,'-------------------------------------------------------------------------------\n');

