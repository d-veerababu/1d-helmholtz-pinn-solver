function parameter = initializeZeros(sz,className)
% Function to initialize biases with zeros
arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter);

end