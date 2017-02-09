function [v, pi] = policyIteration(model)
maxit=100;
tol = 0.001;
nstates = size(model.P,1);
v = zeros(1,nstates);
v_ = zeros(1,nstates);

pi = ones(nstates,1);

it2 = 0;
piStable = false;

while( it2 < 100)
sprintf('one more it')
pi = ones(nstates,1);
    %policy evaluation  
    for state= 1:nstates
        sprintf('here---')
        P = reshape(model.P(state,:,:), size(model.P,2),size(model.P,3));
        R = repmat(model.R(state,:), nstates,1);
        v_old = inf;
        it = 0 ;
        while (abs(v_(state)-v_old) > tol && it < 1)
            sprintf('here')
            it = it+1;
            v_old = v_(state);
            v_(state) = sum(pi(state)*(sum(P.*(R+model.gamma*repmat(v, size(R,2),1)'))));
        end
        it
    end
    
%policy improvement
piStable = true;
for state2= 1:nstates
    
    P2 = reshape(model.P(state2,:,:), size(model.P,2),size(model.P,3));
    R2 = repmat(model.R(state2,:), nstates,1);
    pi_old = pi(state2);
    [A, I] = max(sum(P2.*(R2+model.gamma*repmat(v_, size(R2,2),1)')));
    pi(state2) = I;
        if (pi(state2) ~= pi_old)
            piStable = false;        
        end
        
end
v = v_
it2 = it2+1;

end