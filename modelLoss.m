function [loss,gradients] = modelLoss(parameters,X,U0,freq,c0)
% Make predictions with the initial conditions.
k = 2*pi*freq/c0;
U = model(parameters,X);
phi1 = 1-X;
phi2 = X;

phi_eqv = phi1.*phi2;

% Trial neural network
G = phi1*U0(1)+phi2*U0(2)+phi_eqv.*U;

% Calculate derivatives with respect to X
Gx = dlgradient(sum(G,'all'),X,'EnableHigherDerivatives',true);

% Calculate second-order derivatives with respect to X.
Gxx = dlgradient(sum(Gx,'all'),X,'EnableHigherDerivatives',true);

% Calculate loss.
f = Gxx+k^2*G;
zeroTarget = zeros(size(f),"like",f);
loss = l2loss(f, zeroTarget);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
