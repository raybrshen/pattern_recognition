% parameters: random vectors of the two classes% return: plot figurefunction plot_data(rv_1,rv_2)  figure  % x1-x2 domain  subplot(1,2,1)  scatter(rv_1(:,1),rv_1(:,2),'red')  hold on  scatter(rv_2(:,1),rv_2(:,2),'blue')  hold off  title('distribution in x1-x2 domain')  axis equal  % x1-x3 domain  subplot(1,2,2)  scatter(rv_1(:,1),rv_1(:,3),'red')  hold on  scatter(rv_2(:,1),rv_2(:,3),'blue')  hold off  title('distribution in x1-x3 domain')  axis equalend