function [v, pi] = policyIterationF(model, maxit)

theta = 0.01;
% theta = 1;

% initialize the value function
v = zeros(model.stateCount, 1);

improving=1;
    
for i = 1:maxit
    %% initialize the policy and the new value function
    pi = ones(model.stateCount, 1);
    v_ = zeros(model.stateCount, 1);

    %% policy evaluation
    summ = zeros(size(model.R));
    % perform the Bellman update for each state
    for s = 1:model.stateCount
        for a = 1:size(model.R,2)
            summ(s,a) = (model.P(s,:,a)*(model.R(s,a)+model.gamma*v(:)));
        end
    end
    s = zeros(size(model.R));
    for a = 1:size(model.R,2)
        s(:,a) = pi.*summ(:,a);
    end
    v_ = sum(s,2);


    %% policy improvement
    summ = zeros(size(model.R));
    % perform the Bellman update for each state
    for s = 1:model.stateCount
        for a = 1:size(model.R,2)
            summ(s,a) = (model.P(s,:,a)*(model.R(s,a)+model.gamma*v_(:)));
        end
    end
    for s = 1:model.stateCount
        for a = 1:4
            if summ(s,a)==max(summ(s,:))
                pi_(s) = a;
                break
            end
        end
    end

    %% check / update
    improvement = all(pi_)>=all(pi);
    if ~improvement
        return
    end
    pi = pi_';
    v = v_;

end





