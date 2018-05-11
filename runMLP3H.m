function Y=runMLP3H(X,Wx,Wh,Wy)


[p1 N] = size (X);

bias = 1;

X = [ X;bias*ones(1,N)];



V = Wx*X;
Z = 1./(1+exp(-V));


T = [Z;ones(1,size(Z,2))];
G1 = Wh*T;
zz = 1./(1+exp(-G1));

S = [zz;ones(1,size(zz,2))];
G = Wy*S;
Y = G;%1./(1+exp(-G));

% S = [bias*ones(1,N);ZZ];
% G = Wy*S;
% Y = 1./(1+exp(-G));
