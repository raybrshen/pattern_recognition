% parameters: means, covariance matrices and probabilities of the two classes% return: linear points of discriminant functions on two domainsfunction [dis_1_2,dis_1_3] = discriminant_fn_2d(M1,M2,Sig1,Sig2,P1,P2)  % (x1-x2) domain  m1 = M1([1,2],1);  m2 = M2([1,2],1);  sig1 = Sig1(1:2,1:2);  sig2 = Sig2(1:2,1:2);    A = inv(sig2)-inv(sig1); # dimension 2x2  B = 2*(m1'*inv(sig1)-m2'*inv(sig2)); # dimension 1x2  C = m2'*inv(sig2)*m2-m1'*inv(sig1)*m1-log(det(sig1)/det(sig2))-2*log(P2/P1); # dimension 1x1  dis_1_2 = zeros(41,3);  for i = -20:20    dis_1_2(i+21,1) = i;    fn = @(x)[i,x]*A*[i;x]+B*[i;x]+C;    x0 = -20;    ret = fsolve(fn,x0);    dis_1_2(i+21,2) = ret;    x0 = 20;    ret = fsolve(fn,x0);    dis_1_2(i+21,3) = ret;    % fprintf('%g : %g\n', i, ret)  end    % (x1-x3) domain  m1 = M1([1,3],1);  m2 = M2([1,3],1);  sig1 = Sig1([1,3],[1,3]);  sig2 = Sig2([1,3],[1,3]);    A = inv(sig2)-inv(sig1); # dimension 2x2  B = 2*(m1'*inv(sig1)-m2'*inv(sig2)); # dimension 1x2  C = m2'*inv(sig2)*m2-m1'*inv(sig1)*m1-log(det(sig1)/det(sig2))-2*log(P2/P1); # dimension 1x1    dis_1_3 = zeros(41,3);  for i = -20:20    dis_1_3(i+21,1) = i;    fn = @(x)[i,x]*A*[i;x]+B*[i;x]+C;    x0 = -20;    ret = fsolve(fn,x0);    dis_1_3(i+21,2) = ret;    x0 = 20;    ret = fsolve(fn,x0);    dis_1_3(i+21,3) = ret;    % fprintf('%g : %g\n', i, ret)  end    % display(points)  % line(points(:,1),points(:,2))end