function [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = BPAs(Y, X, id_list, options)
%--------------------------------------------------------------------------
% Budget passive aggressive simple with removal 
%Input:
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
%--------------------------------------------------------------------------
%% some parameters
N= size(Y,1);
ID = id_list;
err_count = 0;
%% options
C = options.C_BPAs; % 1 by default
B = options.Budget;

n_label=options.n_label;
t_tick = options.t_tick ;
%% classifier 
alpha =[];
SV = [];

%% evaluation
mistakes =[];
mistakes_idx =[];
SVs=[];
TMs=[];
size_SV=0;

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    %% compute every label's weight
    if (size_SV==0),
        V=zeros(n_label,1);
    else
        kt=comp_K(X, options, id, SV(1:size_SV));
        V=alpha(:,1:size_SV)*kt;
    end
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
    %% update
    if (l_t>0)
        if size(SV,2)<B;
           size_SV=size_SV+1;
           tau_t = min(C,l_t/2);
           SV=[SV id];
           alpha=[alpha zeros(n_label,1)];          
           alpha(Y(id),size_SV)=tau_t;
           alpha(s_t,size_SV)=-tau_t;
        else            
            Q_star=inf;         
            tau_t = min(C,l_t/2);            
            for r=1:B,
                k_rt =kt(r);
                alpha_r=alpha(:,r);
                beta_t=alpha_r*k_rt;
                beta_t(Y(id))=beta_t(Y(id))+tau_t;
                beta_t(s_t)=beta_t(s_t) - tau_t;
                distance_f_rt = alpha_r'*alpha_r + beta_t'*beta_t - 2*alpha_r'*beta_t*k_rt;                 
                
                f_rt = V - alpha_r*k_rt + beta_t;
                
                idx_n_r=[1:Y(id)-1 Y(id)+1:n_label];
                V_n_r=f_rt(idx_n_r);
                [V_sec_r, idx_sec_r]=max(V_n_r);
                l_rt=max(0, 1-(f_rt(Y(id))-V_sec_r));                                               
                Q_r=0.5*distance_f_rt+C*l_rt;             
                if Q_r<Q_star,
                    Q_star=Q_r;
                    star = r;
                    star_alpha = beta_t;
                end
            end
            alpha(:,star)=star_alpha;
            SV(star)=id;           
        end
    end
    run_time=toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs size_SV];
        TMs = [TMs run_time];
    end
end
run_time = toc;
err_count=err_count/N;
