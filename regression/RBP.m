function [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =RBP(Y, X, id_list, options)
% RBOL_K_M: the random budget online learning algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the column vector of lables, where Y(t) is the lable for t-th instance ;
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j);
%  options:    a struct containing C, tau, rho, sigma, t_tick;
%  id_list:    a random permutation of the 1,2,...,T;
% Output:
%  classifier:  a struct containing SV (the set of idexes for all the support vectors) and alpha (corresponding weights)
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm at a time
%    mistakes:  a vector of online mistake rate
% mistake_idx:  a vector of index, in which every idex is a time and corresponds to a mistake rate in the vector above
%         SVs:  a vector recording the online number of support vectors for every idex in mistake_idx
%         TMs:  a vector recording the online time consumption
%        M_ds:  the number of strong double updating
%        M_dw:  the number of weak double updating
%         M_s:  the number of single updating
%--------------------------------------------------------------------------
%% initialize parameters
t_tick = options.t_tick;
epsl=options.epsl;
eta=options.eta_perceptron;
ID = id_list;
N= size(Y,1);
B      = options.Budget;
%% classifier
alpha = [];
SV = [];
%% evaluation
err_count = 0;
loss_avg=0;
size_SV=0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
loss_v=[];

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    %% compute f_t(x_t)
    if (isempty(alpha)),
        f_t=0;
    else
        k_t = comp_K(X, options, id, SV);
        f_t=alpha*k_t;
    end
    y_t=Y(id);
    delt=y_t-f_t;
    loss_avg=loss_avg+delt^2;
    if (abs(delt)>epsl),
        err_count = err_count + 1;       
        if (size(alpha,2)<B),
           size_SV=size_SV+1;
           alpha = [alpha eta*sign(delt)];
           SV = [SV id];
        else
            perm_t=randperm(B);
            idx=perm_t(1);
            subset=[1:idx-1,idx+1:B];
            alpha=alpha(subset);
            SV = SV(subset);

            alpha = [alpha eta*sign(delt)];
            SV = [SV id];
        end
    end



    %% record performance
    run_time = toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
        loss_v=[loss_v loss_avg/t];
    end
end

run_time = toc;
err_count=err_count/N;
loss_avg=loss_avg/N;