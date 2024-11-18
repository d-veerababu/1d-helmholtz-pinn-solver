function parameter = initializeHe(sz,numIn,className)
% Function to initialize weights using He initialization
arguments
    sz
    numIn
    className = 'single'
end

parameter = sqrt(2/numIn) * randn(sz,className);
parameter = dlarray(parameter);

end