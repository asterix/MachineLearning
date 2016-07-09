function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
#{
sig_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
c_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

err_min = realmax();

for i = 1:size(sig_vec)
   for j = 1:size(c_vec)

   % Train using X,y and selected C and sigma
   model = svmTrain(X, y, c_vec(j), @(x1, x2) gaussianKernel(x1, x2, sig_vec(i)));

   % Predict on cross-validation set
   preds = svmPredict(model, Xval);

   % Check predictions against yval
   err = mean(xor(preds, yval));

   % Find minimum prediction error
   if(err_min > err)
      err_min = err;
      C = c_vec(j);
      sigma = sig_vec(i);
      fprintf('\nMin error = %f. New C & Sigma = [%f %f]\n\n', err, C, sigma);
   endif

   end;
end;
#}
% =========================================================================

end
