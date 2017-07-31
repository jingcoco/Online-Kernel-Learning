function [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] = NyGD(Y, X, id_list, options)
% Nystrom gradient descent
%--------------------------------------------------------------------------
% Input:
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
%% initialize parameters
eta    = options.eta;
B      = options.Budget;
t_tick = options.t_tick;
k = round(options.k * B);
n_label=options.n_label;
%% classifier 
alpha =[];
SV = [];
%% evaluation
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
size_SV=0;
%% loop
flag = 0;
tic
for t = 1:length(ID),
    id = ID(t);
    if(flag==0)
        if (size_SV==0),
        Value=zeros(n_label,1);
        else
        Value=alpha*comp_K(X, options, id, SV);%%%%%%%%%%%%%
        end
        %% predict label
        [Value_max,idx_max]=max(Value);
         hat_y_t=idx_max;
        %% compute the hingh loss and support vector
         idx_n=[1:Y(id)-1 Y(id)+1:n_label];
         Value_n=Value(idx_n);
         [Value_sec, idx_sec]=max(Value_n);
         s_t=idx_sec;
        %% revise the index to be correct
        if s_t>=Y(id),
           s_t=s_t+1;
        end
        l_t=max(0, 1-(Value(Y(id))-Value_sec));
       %% count the number of error and then update
        if hat_y_t~=Y(id),
           err_count=err_count+1;
        end
        if l_t>0,
           if (size_SV<B)
              size_SV=size_SV+1;
              alpha=[alpha zeros(n_label, 1)];
              SV=[SV id];
            %% update Y(id)
              alpha(Y(id),size_SV)=eta;
            %% update idx_sec
              alpha(s_t,size_SV)=-eta;
          else
            k_hat = comp_K1(X, options, SV, SV);
            [V,D] = eigs(k_hat, k);
            
            flag = 1;
            w = alpha*pinv(D^(-0.5)*V');
            k_t = comp_K(X, options, id, SV);
            nx_t = D^(-0.5)*V'*k_t;
            w(Y(id),:)= w(Y(id),:)+eta*nx_t';
            w(s_t,:)= w(s_t,:)-eta*nx_t';
           end
     end
    else%%%%%%%%%%%%%%%%%%%%%%%
        k_t = comp_K(X, options, id, SV);

        nx_t = D^(-0.5)*V'*k_t;

        Value=w*nx_t;
        %% predict label
        [Value_max,idx_max]=max(Value);
         hat_y_t=idx_max;
         %% compute the hingh loss and support vector
         idx_n=[1:Y(id)-1 Y(id)+1:n_label];
         Value_n=Value(idx_n);
         [Value_sec, idx_sec]=max(Value_n);
          s_t=idx_sec;
          %% revise the index to be correct
        if s_t>=Y(id),
           s_t=s_t+1;
        end
        l_t=max(0, 1-(Value(Y(id))-Value_sec));
        %% count the number of error and then update
        if hat_y_t~=Y(id),
        err_count=err_count+1;
        end    
        if l_t>0,
            %% update 
            w(Y(id),:)= w(Y(id),:)+eta*nx_t';
            w(s_t,:)= w(s_t,:)-eta*nx_t';
        end  
   end
   %% record performance
   run_time = toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs size_SV];
        TMs=[TMs run_time];
    end
end

run_time = toc;
err_count=err_count/N;
