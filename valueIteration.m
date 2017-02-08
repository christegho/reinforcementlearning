function [v, pi] = valueIteration(model, maxit)
maxit=20
tol = 0.0001;
nstates = size(model.P,1);
v = zeros(1,nstates);
maxiter = 0;
for state= 1:nstates
    P = reshape(model.P(state,:,:), size(model.P,2),size(model.P,3));
    R = repmat(model.R(state,:), nstates,1);
    v_old = inf;
    it = 0 ;
    while (abs(v(state)-v_old) > tol && it < maxit)
       it = it+1;
       v_old = v(state);
       maxiter = max(it,maxiter);
       v(state) = max(sum(P.*(R+model.gamma*repmat(v, size(R,2),1)')));
    end

end
    