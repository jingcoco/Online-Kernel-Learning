function demo_B(dataset_name,data)
% the impace of budget size to online mistake rate and training time
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------

%% load dataset
load(sprintf('data/%s',dataset_name));
[n,d]       = size(data);
%% set parameters
options.eta    =  0.2;
options.lambda =  1/(n^2);
options.gamma  =  10;
options.t_tick = round(n/15);
options.sigma  = 8;
%options for bpas
options.C_BPAs = 1;
%options for NOGD
options.eta_nogd=0.2;
options.k = 0.2;
%options for FOGD
options.eta_fou=0.002;
options.D=4;
%%  choose a subset of the whole dataset (default: using all)

Y = data(1:n,1);
X = data(1:n,2:d);

B=[50 100 200 300 400 500];

for i=1:size(B,2),
    fprintf(1,'running on the %d-th trial...\n',i);
    options.Budget = B(i);
for j=1:20
        ID_list = ID_all(j,:);
        
    %% run experiments:
        %1. perceptron
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = perceptron(Y,X,options,ID_list);
        err_PE(i,j)             = err_count/n;
        time_PE(i,j)            = run_time;
        %2.ogd
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = OGD(Y,X,options,ID_list);
        err_OGD(i,j)             = err_count/n;
        time_OGD(i,j)            = run_time;     
        %3. RBP
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = RBP(Y,X,options,ID_list);
        err_RP(i,j)             = err_count/n;
        time_RP(i,j)            = run_time;        
        %4
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = forgetron(Y,X,options,ID_list);
        err_FP(i,j)             = err_count/n;
        time_FP(i,j)            = run_time;
         %5. projectron
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = projectron(Y,X,options,ID_list);
        err_PJ(i,j)             = err_count/n;
        time_PJ(i,j)            = run_time;
        %6. projectron++
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = proplus(Y,X,options,ID_list);
        err_PP(i,j)             = err_count/n;
        time_PP(i,j)            = run_time;
        %7
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = BOGD(Y,X,options,ID_list);
        err_BO(i,j)             = err_count/n;
        time_BO(i,j)            = run_time;
        %8         
        %9. BPAs
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = BPAs(Y,X,options,ID_list);
        err_BPAs(i,j)             = err_count/n;
        time_BPAs(i,j)            = run_time;     
         %10. FouGD
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = FouGD(Y,X,options,ID_list);
        err_FGD(i,j)             = err_count/n;
        time_FGD(i,j)            = run_time;
        %11. NysGD
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = NysGD(Y,X,options,ID_list);
        err_NGD(i,j)             = err_count/n;
        time_NGD(i,j)            = run_time;
end
end
save('current');
%% print and plot results
figure
plot(B, mean(err_PE'),'b.-');
hold on
plot(B, mean(err_OGD'),'b-+');
plot(B, mean(err_RP'),'g->');
plot(B, mean(err_FP'),'m-v');
plot(B, mean(err_PJ'),'r-p');
plot(B, mean(err_PP'),'r-x');
plot(B, mean(err_BO'),'r-o');
plot(B, mean(err_BPAs'),'g-*');
plot(B,mean(err_FGD'),'m-x')
plot(B,mean(err_NGD'),'g-o')
legend('Perceptron','OGD','RBP','forgetron','projectron','projectron++','BOGD','BPAs','FouGD','NyGD');
xlabel('Budget Size');
ylabel('Online average rate of mistakes')
grid
%saveas(gcf,'dna_mistake_100','fig')

figure
plot(B, log(mean(time_PE'))/log(10),'b.-');
hold on
plot(B, log(mean(time_OGD'))/log(10),'b-V');
plot(B, log(mean(time_RP'))/log(10),'g->');
plot(B, log(mean(time_FP'))/log(10),'m-v');
plot(B, log(mean(time_PJ'))/log(10),'r-p');
plot(B, log(mean(time_PP'))/log(10),'r-x');
plot(B, log(mean(time_BO'))/log(10),'r-o');
plot(B, log(mean(time_BPAs'))/log(10),'g-*');
plot(B,log(mean(time_FGD'))/log(10),'m-x');
plot(B,log(mean(time_NGD'))/log(10),'m-<');
legend('Perceptron','OGD','RBP','forgetron','projectron','projectron++','BOGD','BPAs','FouGD','NyGD');
xlabel('Budget Size');
ylabel('average time cost (log_{10} t)')
grid