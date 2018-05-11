function [Wx,Wy,Wh,MSE,C]=trainMLP3(p,H,H2,m,mu,X1,D1,epochMax,MSETarget)

% Input parameters:
%   p: Number of the inputs.
%   H: Number of first hidden neurons
%   H2: Number of second hidden neurons
%   m: Number of output neurons
%   mu: Learning-rate parameter
%   X: Input matrix.  X is a (p x N) dimensional matrix, where p is a number of the inputs and N is a training size.
%   D: Desired response matrix. D is a (m x N) dimensional matrix, where m is a number of the output neurons and N is a training size.
%   epochMax: Maximum number of epochs to train.
%   MSETarget: Mean square error target.
%
% Output parameters:
%   Wx: first Hidden layer weight matrix. Wx is a (H x p+1) dimensional matrix.
%   Wh: second Hidden layer weight matrix. Wx is a (H2 x H+1) dimensional matrix.
%   Wy: Output layer weight matrix. Wy is a (m x H2+1) dimensional matrix.
%   MSE: Mean square error vector.

[p1 N] = size(X1);
bias = 1;

X1 = [ X1;bias*ones(1,N) ];

Wx =rand(H,p+1);   %/1000;
                   %/1000;%[0.4,0.45,0.6;0.50,0.55,0.6];
Wh = rand(H2,H+1);
Wy = rand(m,H2+1);

dCwx=zeros(size(Wx));
dCwy=zeros(size(Wy));


%WxAnt = zeros(H,p+1);
%Tx = zeros(H,p+1);
%rand(m,H+1);

%Ty = zeros(m,H+1);
%WyAnt = zeros(m,H+1);
%DWy = zeros(m,H+1);
%DWx = zeros(H,p+1);

MSETemp = zeros(1,epochMax);
C=1;
while(C<epochMax)
    dCwx=zeros(size(Wx));
    dCwy=zeros(size(Wy));
    dCwxh=zeros(size(Wh));
    %for i=1:size(X1,2)
        
        p = randperm(N,1);
        X = X1(:,p);
        D = D1(:,p);
        
        V = Wx*X;
        Z = 1./(1+exp(-V));
        
        %add code
        
        T = [Z;1];
        G1 = Wh*T;
        zz = 1./(1+exp(-G1));
        
        S = [zz;1];
        G = Wy*S;
        Y = G;%1./(1+exp(-G));
        
        E = Y-D;
        
        mse = mean(mean(E.^2));
        MSETemp(C) = mse;
        disp(['epoch = ' num2str(C) ' mse = ' num2str(mse)]);
        if (mse < MSETarget)
            MSE = MSETemp(1:C);
            return
        end
        
        
        df =2;% Y.*(1-Y);
        dGy = df .* E;
        S1=repmat(S(1:end-1,:)',m,1);%own
        dGy1=repmat(dGy,1,H2);%before
        dwy=dGy1.*S1;
        dwy(:,end+1)=dGy;
        
        
        
        
        
        %add
        
        df= S.*(1-S);
        dGxn = df .* (Wy' * dGy);
        T11=repmat(T(1:end-1,:)',H2,1);
        dGx1=repmat(dGxn(1:end-1,:),1,H);
        dwxh=dGx1.*T11;
        dwxh(:,end+1)=dGxn(1:end-1,:);
        
        
        
        df= T.*(1-T);
        dGx = df .* (Wh' * dGxn(1:end-1,1));
        X11=repmat(X(1:end-1,:)',H,1);
        dGx11=repmat(dGx(1:end-1,:),1,p1);
        dwx=dGx11.*X11;
        dwx(:,end+1)=dGx(1:end-1,:);
        
        
        
        Wx=Wx-mu.*(dwx);
        Wh=Wh-mu.*(dwxh);
        Wy=Wy-mu.*dwy;
%         dCwx=dCwx+(dwx/N);
%         dCwxh=dCwxh+(dwxh/N);
%         dCwy=dCwy+(dwy/N);
        
        
    %end
    C=C+1;
%     Wx=Wx-mu.*(dCwx);
%     Wh=Wh-mu.*(dCwxh);
%     Wy=Wy-mu.*(dCwy);
    
end

MSE = MSETemp;
end



