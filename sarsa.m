function [v, pi] = sarsa(model, maxit, maxeps)
v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
% initialize the value function
Q = zeros(model.stateCount, 4);
for i = 1:maxeps,
    % every time we reset the episode, start at the given startState
    s = model.startState;
    a=1;
    for j = 1:maxit,
        % PICK AN ACTION
        p = 0;
        r = rand;

        for s_ = 1:model.stateCount,
            p = p + model.P(s, s_, a);
            if r <= p,
                break;
            end
        end
        
        
        %cumulativeR(i) = cumulativeR(i) + model.gamma*model.R(s,a);   

        % s_ should now be the next sampled state.
        % IMPLEMENT THE UPDATE RULE FOR Q HERE.
        alpha = 0.4;
        
        eps = 0.1; 
        if rand < (1-eps)
            [mQ, a_] = max(Q(s_,:));        
        else
            a_ = randi(4);       
        end    
        
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a)+model.gamma*Q(s_,a_) -Q(s,a));
        % SHOULD WE BREAK OUT OF THE LOOP?
        s=s_;
        a=a_;
        
        if s == model.goalState
            break;
        end 
        
    end
    oldV = v;
    oldPi = pi;
    [A,I] = max(Q');
    pi = I';
    v = A';
    if (oldPi == pi)
        %break;
    end
    
end

      



