function [ID] = create_rand_ID(n, t)
%generate random permutation of training instances
% n - dataset size
% t - number of trials, usually set to 20
ID=[];
for i=1:t,
    ID = [ID; randperm(n)];
end
