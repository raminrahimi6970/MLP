

clear;clc
p =4; %input
H = 5; %first hiden layer    7
H2=3;
m = 3; %output

mu = 0.1;

Iter = 90000;
MSEmin = 1e-12;


load fisheriris
a=find(strcmp(species,'setosa'));
b=find(strcmp(species,'versicolor'));
c=find(strcmp(species,'virginica'));
T=ones(3,150)-2;
T(1,a)=1;
T(2,b)=1;
T(3,c)=1;
x=meas';
x(1,:)=(x(1,:)-min(x(1,:)))./(max(x(1,:))-min(x(1,:)));
x(2,:)=(x(2,:)-min(x(2,:)))./(max(x(2,:))-min(x(2,:)));
x(3,:)=(x(3,:)-min(x(3,:)))./(max(x(3,:))-min(x(3,:)));
x(4,:)=(x(4,:)-min(x(4,:)))./(max(x(4,:))-min(x(4,:)));
X=x;
D=T;
pe = randperm(150,75);
c=1;
for i=1:150
    f=true;
    for j=1:size(pe,2)
        if i==pe(1,j)
            f=false;
        end
    end
    if f==true
        pt(1,c)=i;
        c=c+1;
    end
            
end

XT=X(:,pe);
DT=D(:,pe);
XTE=X(:,pt);
DTE=D(:,pt);


[Wx,Wy,Wh,MSE,C]=trainMLP3(p,H,H2,m,mu,XT,DT,Iter,MSEmin);

semilogy(MSE);



Y = runMLP3H(XTE,Wx,Wh,Wy);



