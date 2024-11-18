function parameters = buildNet(numLayers,numNeurons)

parameters = struct;

% Input layer
sz = [numNeurons 1];
parameters.fc1_Weights = initializeHe(sz,1,"double");
parameters.fc1_Bias = initializeZeros([numNeurons 1],"double");

% Hidden layers
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz,numIn,"double");
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],"double");
end
% Output layer
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,"double");
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],"double");

end
