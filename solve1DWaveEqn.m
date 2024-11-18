function U = solve1DWaveEqn(X,k)

% Initialize solutions.
U = zeros(size(X));

% Loop over x values.
for i = 1:numel(X)
    x = X(i);
    U(i) = cos(k*x)-(csc(k)+cot(k))*sin(k*x);
end

end
