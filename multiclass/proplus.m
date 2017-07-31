function [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =proplus(Y, X, id_list, options)
%--------------------------------------------------------------------------
% aggressive version of projectron 
%Input:
%        Y:    the column vector of lables, where Y(t) is the lable for t-th instance ;
%        X:    features;
%  id_list:    a random permutation of the 1,2,...,T;
%  options:    a struct containing C, rho, sigma, n_lable, t_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm at a time
%    mistakes:  a vector of online mistake rate
% mistake_idx:  a vector of index, in which every idex is a time and corresponds to a mistake rate in the vector above
%         SVs:  a vector recording the online number of support vectors for every idex in mistake_idx
%     size_SV:  the final size of the support vector set
%         TMs:  a vector recording the online time consumption
%--------------------------------------------------------------------------
%% some parameters
N= size(Y,1);
ID = id_list;
err_count = 0;
%% options
n_label=options.n_label;
t_tick = options.t_tick ;
%% classifier
alpha =[];
SV = [];     % SV vectors
%% evaluation
mistakes =[];
mistakes_idx =[];
SVs=[];
TMs=[];
size_SV=0;

%% initialize parameters
B        = options.Budget;%%%%%eta thrhold wheather to add the support vectors
U         =(1/4)*sqrt((B+1)/log(B+1));
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    %% compute every label's weight
    if (size_SV==0),
        V=zeros(n_label,1);
    else
        V =zeros(n_label,1);
        for i=1:n_label,
            k_t= comp_K(X, options, id, SV);
            V(i)=alpha(i,:)*k_t;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SV row, alpha row
        end
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
    if (isempty(alpha))
        if hat_y_t~=Y(id),
            err_count=err_count+1;
            size_SV=size_SV+1;
            alpha=[alpha zeros(n_label,1)];
            alpha(Y(id),size_SV)=1;
            alpha(s_t,size_SV)=-1;
            SV = [SV id];    
            Kid = comp_K(X, options, id, id);
            K_t_inver=1/Kid;
        end
    else%%not empty
        if hat_y_t~=Y(id),%error
            err_count=err_count+1;  
            
            d_star=K_t_inver*k_t;
            norm_delta_t=Kid-k_t'*d_star; 
            if(size_SV<B)
                size_SV=size_SV+1;
                alpha=[alpha zeros(n_label,1)];
                alpha(Y(id),size_SV)=1;
                alpha(s_t,size_SV)=-1; 
                SV = [SV id];
                
                
                temp=zeros(size_SV);
                temp(1:size_SV-1,1:size_SV-1)=K_t_inver;
                d_til=[d_star;-1];
                K_t_inver=temp+d_til*d_til'/norm_delta_t;
            else
               alpha(Y(id),:)=alpha(Y(id),:)+d_star';
               alpha(s_t,:)=alpha(s_t,:)-d_star';
            end

        elseif (l_t>0)%margin error
            d_star=K_t_inver*k_t;
            norm_delta_t=Kid-k_t'*d_star; 
            P=k_t'*K_t_inver*k_t;
            tau_t=min(l_t/P,1);
            beta_t=tau_t*(2*l_t-tau_t*P-2*U*sqrt(norm_delta_t));
            if(beta_t>=0)
               alpha(Y(id),:)=alpha(Y(id),:)+tau_t*d_star';
               alpha(s_t,:)=alpha(s_t,:)-tau_t*d_star';            
            end
        end
    end

    %% record performance%%%%%%
    run_time = toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs size_SV];
        TMs = [TMs run_time];
    end
    
end

run_time = toc;
err_count=err_count/N;