function [a_next, z_next] = fc(w, a, x, f)
    % define the activation function
    %f = @(s) max(0, s);
    %df = @(s) s >= 0;
    %f = @(s) 1 ./ (1 + exp(-s));

    z_next = w * [x; a];
    a_next = f(z_next);
    
end