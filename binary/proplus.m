function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = proplus(Y, X, options, id_list)
% The aggressive version of projectron algorithm, with update on margin error
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
B        = options.Budget;
t_tick   = options.t_tick;
ID       = id_list;

alpha        = [];
SV           = [];
mistakes     = [];
mistakes_idx = [];
SVs          = [];
TMs          =[];

err_count = 0;
U         =(1/4)*sqrt((B+1)/log(B+1));

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
    %% count the number of errors
    hat_y_t = sign(f_t);
    if (hat_y_t==0),
        hat_y_t=1;
    end

    y_t=Y(id);
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end


    %% update
    if (isempty(alpha)),
        if (hat_y_t~=y_t),
            alpha = [alpha y_t];
            SV = [SV id];
            Kid = comp_K(X, options, id, id);
            K_t_inver=1/Kid;
        end
    else
        l_t=1-y_t*f_t;

        if (hat_y_t~=y_t),
            d_star=K_t_inver*k_t;
            norm_delta_t=sqrt(Kid-k_t'*d_star);

%             K_t=K(SV(:),SV(:));
%             power_p_k_t=d_star'*K_t*d_star;

%             eta= (1/(2*U))*(2*l_t-power_p_k_t-0.5);

            if size(SV,2)==B, %norm_delta_t<=eta||
                alpha=alpha+Y(id)*d_star';
            else
                alpha=[alpha y_t];
                SV = [SV id];

                size_SV=size(SV,2);
                temp=zeros(size_SV);
                temp(1:size_SV-1,1:size_SV-1)=K_t_inver;
                d_til=[d_star;-1];
                K_t_inver=temp+d_til*d_til'/(norm_delta_t^2);
            end
        elseif (l_t<1&&l_t>0),
            d_star=K_t_inver*k_t;
            norm_delta_t=sqrt(Kid-k_t'*d_star);

            K_t=comp_K(X, options, SV, SV);
            power_p_k_t=d_star'*K_t*d_star;

            tau_t=min(l_t/power_p_k_t,1);
            beta_t=tau_t*(2*l_t-tau_t*power_p_k_t-2*U*norm_delta_t);
            if(beta_t>=0)
                alpha=alpha+Y(id)*tau_t*d_star';
            end
        end
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
classifier.alpha = alpha;
run_time = toc;