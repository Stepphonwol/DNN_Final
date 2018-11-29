function delta = bc(w, z, delta_next, df, drop)
    % define the activation function
    %f = @(s) 1 ./ (1 + exp(-s)); 
    % define the derivative of activation function
    %df = @(s) f(s) .* (1 - f(s));
    %df = @(s) s > 0;
    delta = df(z) .* (w' * delta_next);
    delta = delta .* drop;
end