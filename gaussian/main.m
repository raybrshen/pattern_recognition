% settings
whiten = 2; % whiten data flag, 1=do-not-whiten, 2=whiten,
pl_data = true; % plot data flag
pl_cvg = true; % plot convergence flag
pl_dcmn = true; % plot discriminant function flag
cv = true; % cross validation flag
pau = false; % pause flag

% basic configuration
d = 3; % data dimension
n1 = 1000; % number of samples for class 1
n2 = 1000; % number of samples for class 2

% a priori of each class
p1 = n1/(n1+n2);
p2 = n2/(n1+n2);

% generate random mean and covariance for two classes
[M1,Sig1] = generate_para(d);
[M2,Sig2] = generate_para(d);
M1 = [4;4;4];
M2 = [-4;-4;-4];

% generate random variables for each class
rv1 = generate_rv(n1,d,M1,Sig1);
rv2 = generate_rv(n2,d,M2,Sig2);

while whiten>0

  if pl_data
    % plot data
    plot_data(rv1,rv2);
    if pau, pause; end
  end

  if pl_cvg
    % plot convergence of the two classes
    step = 5;
    % training with maximum likelihood
    [ml_cvg1,ml_cvg2,ml_m1,ml_m2,ml_cov1,ml_cov2] = train_ml(rv1,rv2,step);
    plot_convergence(ml_cvg1,ml_cvg2,step);
    if pau, pause; end
    % training with bayesian estimation
    [be_cvg1,be_cvg2,be_m1,be_m2] = train_be(rv1,rv2,step,Sig1,Sig2);
    plot_convergence(be_cvg1,be_cvg2,step);
    if pau, pause; end
    % plot discriminant function
    if pl_dcmn
      [dis_1_2,dis_1_3] = discriminant_fn_2d(ml_m1,ml_m2,ml_cov1,ml_cov2,p1,p2);
      plot_data_discriminant(rv1,rv2,dis_1_2,dis_1_3);
      if pau, pause; end
      [dis_1_2,dis_1_3] = discriminant_fn_2d(be_m1,be_m2,Sig1,Sig2,p1,p2);
      plot_data_discriminant(rv1,rv2,dis_1_2,dis_1_3);
      if pau, pause; end
    end
  end

  if cv
  % n-fold cross validation
    fprintf('=> testing with cross validation...\n');
    fold = 10;
    n3 = 200;
    n4 = 200;
    p3 = n3/(n3+n4);
    p4 = n4/(n3+n4);
    rv3 = generate_rv(n3,d,M1,Sig1);
    rv4 = generate_rv(n4,d,M2,Sig2);
    disp(M1);
    disp(M2);
    [acc_ml,acc_be] = cross_validation(true,fold,rv3,rv4,p3,p4,Sig1,Sig2);
  end
  
  whiten = whiten-1;
  if whiten>0
    % calculate diagonalizing matrix
    dm = diagonalizing_matrix(rv1,rv2,Sig1,Sig2);
    % whiten data
    [rv1,rv2] = diagonalize(rv1,rv2,dm);
    % calculate new mean and covariance
    M1 = (mean(rv1))';
    M2 = (mean(rv2))';
    Sig1 = cov(rv1);
    Sig2 = cov(rv2);
  end
  
end # while whiten>0

