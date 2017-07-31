function [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =forgetron(Y, X, id_list, options)
% forgetron: the random budget online learning algorithm
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


Q=0;

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    if t==273,
    a=1;
    end
    
    %% compute f_t(x_t)
    if (isempty(alpha)),
        f_t=0;
    else
        k_t = comp_K(X, options, id, SV);
        f_t=alpha*k_t;
    end
    %% count the number of errors
    y_t=Y(id);
    delt=y_t-f_t;
    loss_avg=loss_avg+delt^2;
    if (abs(delt)>epsl),
        err_count = err_count + 1;        
        SV_t = [SV id];
        alpha_t = [alpha eta*sign(delt)];
		
        if size(SV,2)<B,
            SV=SV_t;
            alpha=alpha_t;
            size_SV=size_SV+1;
        else
            r_t=SV_t(1);
            
            k_tp=comp_K(X, options, r_t, SV_t);
            f_tp=alpha_t*k_tp;
            mu=abs(f_tp);
            delta=abs(alpha(1));
            
            coeA=delta^2-2*delta*mu;
            coeB=2*delta;
            coeC=Q-15/32*err_count;
            if coeA==0,                                       %%             (delta*phi)^2+2*delta*phi(1-phi*mu)<=15/32*err_count;
                phi=max(0, min(1,-coeC/coeB));
            elseif coeA>0,
                if coeA+coeB+coeC<=0,
                    phi=1;
                else
                    phi=(-coeB+sqrt(coeB^2-4*coeA*coeC))/(2*coeA);
                end
            elseif coeA<0,
                if coeA+coeB+coeC<=0,
                    phi=1;
                else
                    phi=(-coeB-sqrt(coeB^2-4*coeA*coeC))/(2*coeA);
                end
            end
            
            alpha=phi*alpha_t;
			psi_t=(delta*phi)^2+2*delta*phi*(1-phi*mu);
            Q=Q+psi_t;
            
            SV=SV_t(2:end);
            alpha=alpha(2:end);
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