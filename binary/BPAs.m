function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = BPAs(Y, X, options, id_list)
% Budget Passive aggressive algorithm, with simple SV removal strategy
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
%--------------------------------------------------------------------------

%% initialize parameters
C = options.C_BPAs; % 1 by default
B = options.Budget;

t_tick       = options.t_tick;
alpha        = [];
SV           = [];
ID           = id_list;
err_count    = 0;
mistakes     = [];
mistakes_idx = [];
SVs          = [];
TMs          = [];

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    if (isempty(alpha)), % init stage
        f_t = 0;
    else
        k_t = comp_K(X, options, id, SV);
        f_t=alpha*k_t;
    end
    l_t = max(0,1-Y(id)*f_t);   % hinge loss
    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    % count accumulative mistakes
    y_t=Y(id);
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end

    if (l_t>0)
        if size(SV,2)<B;
            % update
            s_t=1;
            tau_t = min(C,l_t/s_t);
            alpha = [alpha Y(id)*tau_t];
            SV = [SV id];
        else
            Q_star=inf;
            alpha_star=alpha;
            SV_star   = SV;
            
            %% a equivalent form of f_t
%             alpha_equi = [alpha, 0];
%             SV_equi    = [SV, id];
            
            k_tt=s_t;
            tau_t = min(C,l_t/k_tt);
            
%             k_rv = comp_K(X, options, SV, id);
            
            for r=1:B,
                k_rt = k_t(r);
%                 k_rr = comp_K(X, options, SV(r), SV(r));
%                 k_rr = 1;
                alpha_r=alpha(r);

                beta_t=alpha_r*k_rt + tau_t*y_t;                               
                distance_f_rt = alpha_r^2 + beta_t^2 - 2*alpha_r*beta_t*k_rt;
                 
                f_rt = f_t - alpha_r*k_rt + beta_t;
                l_rt=max(0, 1-y_t*f_rt);

                Q_r=0.5*distance_f_rt+C*l_rt;
%                 alpha_r = alpha;
%                 alpha_r(r) = beta_t;
%                 SV_r = SV;
%                 SV_r(r) = id;
                
                if Q_r<Q_star,
                    Q_star=Q_r;
                    star = r;
                    star_alpha = beta_t;
%                     alpha_star = alpha;
%                     alpha_star(r) = beta_t;
%                     SV_star = SV;
%                     SV_star(r) = id;
%                     SV_star=SV_r;
%                     alpha_star=alpha_r;
                end
            end
            
            alpha_star = alpha;
            alpha_star(star) = star_alpha;
            SV_star = SV;
            SV_star(star) = id;
            alpha=alpha_star;
            SV   =SV_star;
            
        end
    end
    run_time=toc;
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end
classifier.SV = SV;
classifier.alpha = alpha;
run_time = toc;
