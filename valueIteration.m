function [v, pi, normV, it] = valueIteration(model, maxit)

tol = 0.001;
nstates = size(model.P,1);
v = zeros(1,nstates);
v_ = zeros(1,nstates);
pi = zeros(nstates,1);
maxiter = 0;
diff = inf;
normV = zeros(1,1);
it=0;
while (diff > tol && it < maxit)
    maxiter = max(it,maxiter);

    for state= 1:nstates
        P = reshape(model.P(state,:,:), size(model.P,2),size(model.P,3));
        R = repmat(model.R(state,:), nstates,1);
        [A, I] = max(sum(P.*(R+model.gamma*repmat(v, size(R,2),1)')));
        v_(state) = A;
        pi(state) = I;
    end
    it = it+1;
    diff = norm(v-v_,3);
    normV = [normV, norm(v_,3)];
    v = v_
end
maxiter
end
    