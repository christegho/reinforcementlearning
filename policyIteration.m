function [v, pi, normV, itTotal] = policyIteration(model,maxit)
tol = 0.001;
nstates = size(model.P,1);
v = zeros(1,nstates);
v_ = zeros(1,nstates);

pi = ones(nstates,1);
maxiter = 0
it2 = 0;
piStable = false;
normV = zeros(1,1);
itTotal = zeros(1,1);

while( it2 < maxit && ~piStable)

    %policy evaluation  
    diff = inf;
    it=0;
    while (diff > tol && it < maxit)
        maxiter = max(it,maxiter);

        for state= 1:nstates
            P = reshape(model.P(state,:,:), size(model.P,2),size(model.P,3));
            R = repmat(model.R(state,:), nstates,1);
            pi_prob = zeros(1,size(R,2));
                pi_prob(pi(state)) = 1;
            v_(state)  = sum(pi_prob.*(sum(P.*(R+model.gamma*repmat(v, size(R,2),1)'))));

        end
        it = it+1;
        diff = norm(v-v_,3);
        normV = [normV, norm(v_,3)];
        v = v_
    end
    itTotal = [itTotal, it-1];
    %policy improvement
    piStable = true;
    for state2= 1:nstates

        P2 = reshape(model.P(state2,:,:), size(model.P,2),size(model.P,3));
        R2 = repmat(model.R(state2,:), nstates,1);
        pi_old = pi(state2);
        [A, I] = max(sum(P2.*(R2+model.gamma*repmat(v, size(R2,2),1)')));
        pi(state2) = I;
            if (pi(state2) ~= pi_old)
                piStable = false;        
            end

    end
    it2 = it2+1;

end