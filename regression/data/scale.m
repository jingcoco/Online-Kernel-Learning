
[n,m]=size(data);
f_max=max(data);
f_min=min(data);
data=(data-repmat(f_min,[n,1]))./(repmat((f_max-f_min),[n,1]));
clear f_max
clear f_min
clear m
clear n