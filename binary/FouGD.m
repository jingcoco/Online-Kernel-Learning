function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = FouGD(Y, X, options, id_list)
% Fourier online gradient algorithm
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
eta    = options.eta_fou;
t_tick = options.t_tick;

ID = id_list;
err_count = 0;

SV = [];
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
options.Budget = options.Budget*options.D;%%%%%%%%%%%%
w = zeros(1, options.Budget*2);
u = (1/options.sigma)*randn(size(X,2),options.Budget);

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    x_t = X(id,:);
    nx_t = [cos(u'*x_t'); sin(u'*x_t')];
    %% compute f_t(x_t)
    f_t=w*nx_t;
    %% count the number of errors
    hat_y_t = sign(f_t);
    if (hat_y_t==0),
        hat_y_t=1;
    end

    y_t=Y(id);
%     l_t = max(0,1-Y(id)*f_t);   % hinge loss
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end
    
    if Y(id)*f_t < 1
%         s_t = norm(nx_t)^2;
%         tau_t = min(C,l_t/s_t);
        w = w + eta*y_t*nx_t';%%%%%%%%%%%
        SV = [SV id];
    end



    %% record performance
    run_time = toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end

classifier.SV = SV;
classifier.w = w;
run_time = toc;


