% Program to solve 1-D Helmholtz equation in a uniform duct 

clc;
clear all;

%% Generate training data
freq = 2000;                     % Frequency (Hz)
c0 = 340;                       % Speed of sound in air (m/s)
k = 2*pi*freq/c0;               % Wavenumber (1/m)

x0BC1 = 0;                      % Left boundary (m)
x0BC2 = 1;                      % Right boundary (m)

u0BC1 = 1;                      % Left boundary value (Pa)
u0BC2 = -1;                     % Right boundary value (Pa)

X0 = [x0BC1 x0BC2];     
U0 = [u0BC1 u0BC2];

numInternalCollocationPoints = 14000;

pointSet = sobolset(1);                   % Base-2 digital sequence that fills space in a highly uniform manner
points = net(pointSet,numInternalCollocationPoints);    % Generates quasirandom point set

dataX = points;         % Creates random x-data points between 0 and 1

%% Define deep learning model
numLayers = 5;
numNeurons = 90;

parameters = buildNet(numLayers,numNeurons);

%% Specify optimization options
options = optimoptions("fmincon", ...
    HessianApproximation="lbfgs", ...
    MaxIterations=14000, ...
    MaxFunctionEvaluations=14000, ...
    OptimalityTolerance=1e-3, ...
    SpecifyObjectiveGradient=true, ...
    Display='iter');

%% Train network
start = tic;

[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);

%% Convert the variables into deep-learning variables
X = dlarray(dataX,"BC");
X0 = dlarray(X0,"CB");
U0 = dlarray(U0,"CB");


objFun = @(parameters) objectiveFunction(parameters,X,U0,parameterNames,parameterSizes,freq,c0);

parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);

parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

toc(start)

%% Evaluate model accuracy
numPredictions = 500;
XTest = linspace(x0BC1,x0BC2,numPredictions);

dlXTest = dlarray(XTest,'CB');
U = model(parameters,dlXTest);

% Space functions
phi1_test = 1-dlXTest;
phi2_test = dlXTest;
phi_eqv_test = phi1_test.*phi2_test;

% Predicted solution 
dlUPred = phi1_test*U0(1)+phi2_test*U0(2)+phi_eqv_test.*U;

% True solution
UTest = solve1DWaveEqn(XTest,k);

% Calculate error.
err = norm(extractdata(dlUPred) - UTest) / norm(UTest);

f1 = figure;
% Plot predicted solution
plot(XTest,extractdata(dlUPred),'-','LineWidth',2);

% Plot true solution
hold on
plot(XTest, UTest, '--','LineWidth',2)
hold off

xlabel('x (m)','FontSize',14,'FontWeight','bold')
ylabel('Acoustic Pressure (Pa)','FontSize',14,'FontWeight','bold')
title("Frequency = " + freq + " Hz;" + " Error = " + gather(err));

legend('Predicted solution','True solution')

