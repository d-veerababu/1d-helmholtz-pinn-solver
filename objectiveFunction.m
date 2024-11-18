function [loss,gradientsV] = objectiveFunction(parametersV,X,U0,parameterNames,parameterSizes,freq,c0)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model loss and gradients.
[loss,gradients] = dlfeval(@modelLoss,parameters,X,U0,freq,c0);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end
