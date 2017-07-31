function [k] = comp_K(X, options, id, SV)
%compute kernels
%X---------the feature values of all instances
%options---the parameter settings, gaussian kernel width

   gid = sum(X(id,:).*X(id,:),2);
   gsv = sum(X(SV,:).*X(SV,:),2);
   gidsv = X(id,:)*X(SV,:)';
   k = exp(-(repmat(gid',length(SV),1) + repmat(gsv,1,length(id)) - 2*gidsv')/(2*options.sigma^2));%%%%%%%%%%
end

