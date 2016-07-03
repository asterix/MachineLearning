function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% m x 1 vector of ht(x)
htx = sigmoid(X * theta);

% Now set theta(1) = 0, so that it is not used in regularization
regt = theta;
regt(1) = 0;

% Compute cost + regularize
J = (-1 / m) * (sum(y.* log(htx) + (1 - y).* log(1 - htx)));
J = J + (lambda / (2 * m)) * sum(regt.^2);

% Compute gradient + regularize
grad = (1 / m) * (X' * (htx - y));
grad = grad + (lambda / m) * regt;

% =============================================================

end
