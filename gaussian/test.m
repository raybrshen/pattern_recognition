% parameters: random variables of the two classes, discriminant function parameters%             flag of printing result% return: true positive and accuracy of the two classesfunction [tp,acc] = test(class_1,class_2,A,B,C,pr)  % testing class 1  [m,n] = size(class_1);  cnt = 0;  for i = 1:m    x = class_1(i,:);    cls = x*A*x'+B*x'+C;    if cls>0      cnt = cnt+1;    end  end  tp1 = cnt;  acc1 = double(cnt)/double(m);  if pr    fprintf('class-1 true positives: %d\n', tp1);    fprintf('class-1 test accuracy rate: %g\n', acc1);  end    % testing class 2  [m,n] = size(class_2);  cnt = 0;  for i = 1:m    x = class_2(i,:);    cls = x*A*x'+B*x'+C;    if cls<0      cnt = cnt+1;    end  end  tp2 = cnt;  acc2 = double(cnt)/double(m);  if pr    fprintf('class-2 true positives: %d\n', tp2);    fprintf('class-2 test accuracy rate: %g\n', acc2);  end    % combine results of two classes  tp = tp1+tp2;  acc = double(tp)/double(2*m);end