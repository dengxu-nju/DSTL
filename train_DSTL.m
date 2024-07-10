function [Y] = train_DSTL(X, gt, param ,k)

%% Normalized data
num_view = length(X);%number of views/features
N = numel(gt);% number of samples
dk=cell(1, num_view);

for i=1:num_view
    X{i} = NormalizeData(X{i});
    dk{i} = size(X{i},1);
    num{i} = size(X{i}, 2);
end

%% parameters
lambda1  = param.lambda1;
lambda2  = param.lambda2;
lambda3  = param.lambda3;

%% initialization
for i = 1:num_view   
    di = size(X{i},1);
    H{i} = zeros(k,N);
    S{i} = zeros(k,N);
    W{i} = zeros(di,k);
    C{i} = zeros(k,k);
    Y = zeros(k,N);
end

MAX_iter = 20;
flag = 1;
iter = 0;

%% update
while flag
    iter = iter + 1;
    Ypre=Y;

    if mod(iter, 10)==0
        fprintf('%d..',iter);
    end

    % W_i
    for i=1:num_view
        [U,~,V] = svd(X{i} *(H{i}+S{i})','econ');
        W{i} = U*V';
        clear  U V;
    end  
       
    % C_i
    for i=1:num_view
        [U,~,V] = svd(H{i}*Y','econ');
        C{i} = U*V';
        clear  U V;
    end  
    
    % S_i
    for i=1:num_view
       S{i} = prox_l1(W{i}'*X{i}-H{i}, lambda1/2);
    end
    
    % H_i
    for i=1:num_view
        OO{i} = W{i}'*X{i}-S{i};
        TT{i} = C{i}*Y;
    end 
    OO_tensor = cat(3, OO{ : , : });
    TT_tensor = cat(3, TT{ : , : });
    OOv = OO_tensor(:);
    TTv = TT_tensor(:);
    [Lv, ~] = wshrinkObj(1/(1+lambda3)*OOv + lambda3/(1+lambda3)*TTv, lambda2/(2*(1+lambda3)), [k, N, num_view], 0, 1);
    H_tensor = reshape(Lv, [k, N, num_view]);
    for i=1:num_view
        H{i} = H_tensor(:,:,i);
    end 
    
    % Y
    tempY = 0;
    for i = 1:num_view
        tempY = tempY + C{i}'*H{i};
    end
    Y = zeros(size(tempY));
    for i = 1:size(tempY, 2)
        Y(:,i) = EProjSimplex_new(tempY(:,i));
    end
    
    % CONVERGENCE
    converge(iter)=norm(Y-Ypre,'fro')^ 2/norm(Y,'fro')^ 2;
    if (iter>1) && (converge(iter)<1e-4 || iter>MAX_iter )
         flag = 0;
    end

end