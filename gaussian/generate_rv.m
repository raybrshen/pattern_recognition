% parameters: number of vectors, dimension, mean, covariance matrix
% return: normalized random vectors
function rv = generate_rv(n,d,M,Sigma)
  % generate normal random vectors with 0-mean 1-variance
  nrv = zeros(n,d);
  for i = 1:12
    % uniform r.v. are 0.5-mean 1/12-variance
    uniform_rv = rand(n,d);
    % add 12 times to get 6-mean 1-variance
    nrv = nrv.+uniform_rv;
  end
  % subtract 6 to get 0-mean 1-variance
  nrv = nrv.-6;
  % compute eigenvector and eigenvalue of covariance matrix
  [eig_vec,eig_val] = eig(Sigma);
  % process normal random vectors to be 0 mean and Sigma covariance
  rv = (eig_vec*(eig_val^(1/2))*(nrv'))';
  % add mean to get objective random vectors
  rv = rv.+M';
end
