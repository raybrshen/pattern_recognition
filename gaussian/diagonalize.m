% parameters: random vectors and the diagonalizing matrix
% return: whitened data of two classes
function [wt_1,wt_2] = diagonalize(rv_1,rv_2,dm)
  % simultaneously diagonalize two distributions
  wt_1 = (dm*(rv_1'))';
  wt_2 = (dm*(rv_2'))';
end