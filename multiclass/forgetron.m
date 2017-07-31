function [err_count, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =forgetron(Y, X, id_list, options)
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
%% options
n_label=options.n_label;
t_tick = options.t_tick ;
B      = options.Budget;
%% classifier
alpha =zeros(n_label,B+1);   % weight matrix
SV = zeros(n_label,B+1);     % SV matrix
svLength=zeros(n_label,1); % the number of sv for every prototype
SV_list=[];
%% evaluation
mistakes =[];
mistakes_idx =[];
SVs=[];
TMs=[];
size_SV=0;

Q=0;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
            size_svi = svLength(i);
            if size_svi~=0,
                SV_i=SV(i,1:size_svi);
                alpha_i=alpha(i,1:size_svi);
                f_i=alpha_i*comp_K(X, options, id, SV_i);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SV row, alpha row
                V(i)=f_i;
            end
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
    if hat_y_t~=Y(id),
        err_count=err_count+1;
        size_SV=size_SV+1;
        SV_list=[SV_list id];
        gamma_t=1;        
       %% update prototype for label Y(id)
        alpha(Y(id),svLength(Y(id))+1)=gamma_t;
        SV(Y(id),svLength(Y(id))+1)= id;
        svLength(Y(id))=svLength(Y(id))+1;
       %% update prototype for label s_t
        alpha(s_t,svLength(s_t)+1)=-gamma_t;
        SV(s_t,svLength(s_t)+1)=id;
        svLength(s_t)=svLength(s_t)+1;
       %% forgetron
        if(size_SV>B)
         %%shrinking
           r_t=SV_list(1);
           size_svi = svLength(Y(r_t));
           SV_i=SV(Y(r_t),1:size_svi);
           alpha_i=alpha(Y(r_t),1:size_svi);
           mu=alpha_i*comp_K(X, options, r_t, SV_i);
           delta=abs(alpha_i(1));
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
           alpha=phi*alpha;
		   psi_t=(delta*phi)^2+2*delta*phi*(1-phi*mu);
           Q=Q+psi_t;
           %%delete
           [x_find,y_find]=find(SV==r_t);
            for i=1:length(x_find)
               SV(x_find(i),:)=[SV(x_find(i),[1:y_find(i)-1,y_find(i)+1:(B+1)]),0];
               alpha(x_find(i),:)=[alpha(x_find(i),[1:y_find(i)-1,y_find(i)+1:(B+1)]),0];
               svLength(x_find(i))=svLength(x_find(i))-1;  
            end
            SV_list=SV_list(2:end);
            size_SV=size_SV-1;
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