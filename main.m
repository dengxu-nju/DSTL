clear; close all;

addpath ./ClusteringMeasure
addpath ./twist
path = './data/';

%%
load ./data/BBCSport.mat; 
param.lambda1 = 1e0;% grid search
param.lambda2 = 5e0;% grid search
param.lambda3 = 1e-4;%fixed

gt = Y;
k = length(unique(Y));
cls_num = numel(unique(Y));
perf = [];
tic
[YY] = train_DSTL(X, Y, param, k);
YY = NormalizeData(YY);
for kk=1:10
  [Clus] = litekmeans(YY', cls_num,'MaxIter',100, 'Replicates',10);
  if kk ==1
  toc
  end
  ACC = Accuracy(Clus, gt);
  [~,NMI,~] = compute_nmi(Clus, gt);
  [f, ~,~]=compute_f(gt, Clus);
  [ARI,~,~]=RandIndex(gt, Clus);
  [~,~, PUR] = purity(gt,Clus);
  pp = [ACC NMI PUR ARI f];
  perf = [perf; pp];
end
mean(perf)
std(perf)

