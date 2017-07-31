function [ID] = create_rand_ID(n, t)
% n - dataset size
% t - number of trials
%
ID=[];
for i=1:t,
    ID = [ID; randperm(n)];
end
