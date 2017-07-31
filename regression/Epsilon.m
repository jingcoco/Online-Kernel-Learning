function [err_count,loss_avg,loss_v, run_time, mistakes, mistakes_idx, SVs, size_SV, TMs] =Epsilon(Y, X, id_list, options)
% Norma: an online learning algorithm for regression with
% epsilon-insensitive loss function
%--------------------------------------------------------------------------
%% initialize parameters
t_tick = options.t_tick;
epsl=options.epsl;
eta=options.Epsilon_eta;
ID = id_list;
N= size(Y,1);

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
    %%
    y_t=Y(id);
    delt=y_t-f_t;
    loss_avg=loss_avg+delt^2;
    %% update
    alpha=(1-options.lambda*eta)*alpha;
    if (abs(delt)>epsl),
        err_count = err_count + 1;
        size_SV=size_SV+1;

        alpha = [alpha eta*sign(delt)];
        SV = [SV id];
        epsl=epsl+(1-options.nu)*eta;
    else
        epsl=epsl-options.nu*eta;
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
