function [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = FouGD(Y, X, id_list, options)
% Fourier Gradient Descent
%--------------------------------------------------------------------------
% Input:
%        Y:    the column vector of lables, where Y(t) is the lable for t-th instance ;
%        X:    features
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

%% some parameters
N= size(Y,1);
ID = id_list;
err_count = 0;
%% options
n_label=options.n_label;
eta    = options.eta_fou;
t_tick = options.t_tick;
options.Budget=round(options.rou_f*options.Budget);
%% classifier

w =zeros(n_label,options.Budget*2);%%%%%%%%budget is D
u = (1/options.sigma)*randn(size(X,2),options.Budget);

%% evaluation
size_SV=0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];



%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    x_t = X(id,:);
    nx_t = [cos(u'*x_t'); sin(u'*x_t')];%%%column
    %% compute f_t(x_t)
    V=w*nx_t;
    %% predict label
    [V_max,idx_max]=max(V);
    hat_y_t=idx_max;
    %% compute the hingh loss and support vector
    idx_n=[1:Y(id)-1 Y(id)+1:n_label];
    V_n=V(idx_n);
    [V_sec, idx_sec]=max(V_n);
    s_t=idx_sec;
    %% revise the index to be correct
    if s_t>=Y(id),
        s_t=s_t+1;
    end
    l_t=max(0, 1-(V(Y(id))-V_sec));
    %% count the number of error and then update
    if hat_y_t~=Y(id),
        err_count=err_count+1;
    end    
    if l_t>0,
        %% update 
        w(Y(id),:)= w(Y(id),:)+eta*nx_t';
        w(s_t,:)= w(s_t,:)-eta*nx_t';
    end 
    
    %% record performance
    run_time = toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs 0];
        TMs=[TMs run_time];
    end
end
run_time = toc;
err_count=err_count/N;

