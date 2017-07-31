function  Experiment_k
%evaluate the k of nygd
%--------------------------------------------------------------------------
clear all
time_run=8;
k=0.025:0.025:0.5;
%% dna
dna=1
load('data\dna');
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);

options.C=10;
options.rho=0.2;% rho \in [0,1]
options.eta    =  0.5;%learning rate
options.Budget=200;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;


for j=1:size(k,2)
options.k=k(j);
for i=1:time_run,
    ID = ID_list(i,:);
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y, X, ID, options);
    err_FG(i) = err_count;
    time_FG(i) = run_time;
end
time_dna(j)=log(mean(time_FG))/log(10);
err_dna(j)=mean(err_FG);
end
save('current_k.mat')

%% satimage
satimage=1
load('data\satimage');
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);

options.C=10;
options.rho=0.2;% rho \in [0,1]
options.eta    =  0.5;%learning rate
options.Budget=200;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;


for j=1:size(k,2)
options.k=k(j);
for i=1:time_run,
    ID = ID_list(i,:);
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y, X, ID, options);
    err_FG(i) = err_count;
    time_FG(i) = run_time;

end
time_sat(j)=log(mean(time_FG))/log(10);
err_sat(j)=mean(err_FG);
end
save('current_k.mat')
%% usps
usps=1
load('data\usps');
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);

options.C=10;
options.rho=0.2;% rho \in [0,1]
options.eta    =  0.5;%learning rate
options.Budget=200;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;

for j=1:size(k,2)
options.k=k(j);
for i=1:time_run,
    ID = ID_list(i,:);
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y, X, ID, options);
    err_FG(i) = err_count;
    time_FG(i) = run_time;

end
time_usps(j)=log(mean(time_FG))/log(10);
err_usps(j)=mean(err_FG);
end
save('current_k.mat')
%% letter
letter=1
load('data\letter');
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);

options.C=10;
options.rho=0.2;% rho \in [0,1]
options.eta    =  0.5;%learning rate
options.Budget=200;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;

for j=1:size(k,2)
options.k=k(j);
for i=1:time_run,
    ID = ID_list(i,:);
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y, X, ID, options);
    err_FG(i) = err_count;
    time_FG(i) = run_time;

end
time_letter(j)=log(mean(time_FG))/log(10);
err_letter(j)=mean(err_FG);
end
save('current_k.mat')
%% shuttle
shuttle=1
load('data\shuttle');
[n,d]= size(data);
Y=data(:,1);
X=data(:,2:end);

options.C=10;
options.rho=0.2;% rho \in [0,1]
options.eta    =  0.5;%learning rate
options.Budget=200;     %%budget....D
options.k=0.2;
options.sigma=8;     %sigma: kernel width
options.n_label=max(Y);    %the number of possible label set
options.t_tick=round(n/10);   %the number of possible label set
options.lambda =  10^(-5);
options.gamma  =  inf;
options.C_BPAs = 1;

for j=1:size(k,2)
options.k=k(j);
for i=1:time_run,
    ID = ID_list(i,:);
    % 5. FouGD
    [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y, X, ID, options);
    err_FG(i) = err_count;
    time_FG(i) = run_time;

end
time_shuttle(j)=log(mean(time_FG))/log(10);
err_shuttle(j)=mean(err_FG);
end
save('current_k.mat')

%% draw graph
figure
plot(k,time_dna,'bd-');
hold on
plot(k,time_sat,'b->');
plot(k,time_usps,'b-v');
plot(k,time_letter,'b-o');
plot(k,time_shuttle,'b-*');
legend('dna','satimage','usps','letter','shuttle');
xlabel('k');
ylabel('average time cost (log_{10} t)')
grid

figure
plot(k,err_dna,'bd-');
hold on
plot(k,err_sat,'b->');
plot(k,err_usps,'b-v');
plot(k,err_letter,'b-o');
plot(k,err_shuttle,'b-*');
legend('dna','satimage','usps','letter','shuttle');
xlabel('k');
ylabel('Online average rate of mistakes')
grid


