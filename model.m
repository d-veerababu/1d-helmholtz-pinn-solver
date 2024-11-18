function U = model(parameters,X)

numLayers = numel(fieldnames(parameters))/2;

% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
U = fullyconnect(X,weights,bias);

% sin and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    U = sin(U);

    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    U = fullyconnect(U, weights, bias);
end

end
