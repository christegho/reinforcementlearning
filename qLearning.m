function [v, pi, cumulativeR,  itEps, epsIt] = qLearning(model, maxit, maxeps)
v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
% initialize the value function
Q = zeros(model.stateCount, 4);
cumulativeR = zeros(maxeps,1);
itEps = [];
epsIt = [];
for i = 1:maxeps,
    % every time we reset the episode, start at the given startState
    s = model.startState;
    eps = 1/i; 
    if rand < (1-eps)
    	[mQ, a] = max(Q(s,:));        
    else
    	a = randi(4);       
    end 
    for j = 1:maxit,
        % PICK AN ACTION
        p = 0;
        r = rand;
        
        %Choose a from s using policy derived from Q
        if rand < (1-eps)
            [mQ, a] = max(Q(s,:));        
        else
            a = randi(4);       
        end
        
        %take action a, observe r and s_
        for s_ = 1:model.stateCount,
            p = p + model.P(s, s_, a);
            if r <= p,
                break;
            end
        end
        
        
           

        %update rule for q
        alpha = 0.2;
        
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a)+max(model.gamma*Q(s_,:)) -Q(s,a));
        
        s=s_;
        cumulativeR(i) = cumulativeR(i) + model.R(s,a);
        epsIt = [epsIt, i];
        if s == model.goalState
            break
        end 
        
        
    end
    itEps = [itEps ,j];
    
    oldV = v;
    oldPi = pi;
    [A,I] = max(Q');
    pi = I';
    v = A';
    if (oldPi == pi)
        %break;
    end
    
end

      



