function [k] = comp_K(X, options, id, SV)
   gid = X(id,:)*X(id,:)';
   gsv = sum(X(SV,:).*X(SV,:),2);%%%%%%%%% column
   gidsv = X(id,:)*X(SV,:)';%%%%%%%%%%row
   k = exp(-(gid + gsv - 2*gidsv')/(2*options.sigma^2));%%%%%%%%%%column
end

