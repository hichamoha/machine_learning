%% DATA

clearvars, close all

% class 1
% dog

dn = 100;
dm = [1,1];
ds = 0.3;
rho = 0.9;
%Xdog = ds*randn(2,dn) + dm(:)*ones(1,dn);
Xdog = mvnrnd(dm,ds^2*[1,rho;rho,1],dn)'
Tdog = [1;0]*ones(1,dn);



% class 2
% cat

cn = 80;
cm = [1.8,1];
cs = 0.3;
rho = 0.9;
%Xcat = cs*randn(2,cn) + cm(:)*ones(1,cn);
Xcat = mvnrnd(cm,cs^2*[1,rho;rho,1],cn)'
Tcat = [0;1]*ones(1,cn);


% join data

T = [Tdog Tcat];
X = [Xdog Xcat];



% RECONSTRUCTION GRID


x1min = min(X(1,:));
x1max = max(X(1,:));
x2min = min(X(2,:));
x2max = max(X(2,:));

x1min = -1;
x1max = 3;
x2min = -1;
x2max = 3;

ngrid = 100;
Xgrid = zeros(2,ngrid^2);
x1grid = linspace(x1min,x1max,ngrid);
x2grid = linspace(x2min,x2max,ngrid);



nind = 1;
for n1 = 1:ngrid
    
    for n2 = 1:ngrid
   
        Xgrid(:,nind) = [x1grid(n1),x2grid(n2)]';
        
        nind = nind + 1;
    end
     
end

% PLOT DATA


figure(1), clf, hold on,
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
legend('Dog','Cat','Hippo')
xlim([x1min x1max])
ylim([x2min x2max])
axis equal
hold off



%% Estimate naive

mdog = mean(Xdog,2);
mcat = mean(Xcat,2);
w = -(mdog - mcat)/sqrt(3)/norm(mdog-mcat,2);


y = w'*Xgrid;
ymat = reshape(y,ngrid,ngrid);

m = (mcat+mdog)/2;
moff = [0;0];

t = linspace(-5,5,100);
roff = moff*ones(size(t)) - w*t;
ron = m*ones(size(t)) - w*t;


%Mdog = mdog*ones(1,dn);
%Mcat = mcat*ones(1,cn);
for k = 1:dn
    rdog(:,k) = moff - ((moff-Xdog(:,k))'*w)/(w'*w)*w;
end
for k = 1:cn
    rcat(:,k) = moff - (moff-Xcat(:,k))'*w/(w'*w)*w;
end
%%


figure(2), clf
subplot(3,1,1:2), hold on,
plot(ron(1,:),ron(2,:),':','LineWidth',2)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
contour(x1grid,x2grid,ymat,10)
plot(roff(1,:),roff(2,:),':','LineWidth',2)
plot(rcat(1,:),rcat(2,:),'o')
plot(rdog(1,:),rdog(2,:),'o')
plot(mcat(1),mcat(2),'*')
plot(mdog(1),mdog(2),'*')
legend('r = m - tw', 'Dog','Cat','y = w^Tx','Location','best')
axis equal
xlim([x1min x1max])
ylim([x2min x2max])
hold off


subplot(313)
hold on
histogram(w'*Xdog,15)
histogram(w'*Xcat,15)
legend('y_{dog}','y_{cat}')
hold off



%% Estimation with Fisher

SB = (mdog-mcat)*(mdog-mcat)'
SW = (Xdog - mdog*ones(1,dn))*(Xdog - mdog*ones(1,dn))' + ...
        (Xcat - mcat*ones(1,cn))*(Xcat - mcat*ones(1,cn))'

[W,D] = eig(inv(SW)*SB);
[lambdamax,ind] = max(diag(D));
w = W(:,ind);

t = linspace(-3,2,100);
ron = m*ones(size(t)) - w*t;
roff = moff*ones(size(t)) - w*t;

y = w'*Xgrid;
ymat = reshape(y,ngrid,ngrid);

for k = 1:dn
    rdog(:,k) = moff - ((moff-Xdog(:,k))'*w)/(w'*w)*w;
end
for k = 1:cn
    rcat(:,k) = moff - (moff-Xcat(:,k))'*w/(w'*w)*w;
end

figure(3), clf
subplot(3,1,1:2), hold on,
plot(ron(1,:),ron(2,:),':','LineWidth',2)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
contour(x1grid,x2grid,ymat,10)
plot(roff(1,:),roff(2,:),':','LineWidth',2)
plot(rcat(1,:),rcat(2,:),'o')
plot(rdog(1,:),rdog(2,:),'o')
plot(mcat(1),mcat(2),'*')
plot(mdog(1),mdog(2),'*')
legend('r = m - tw', 'Dog','Cat','y = w^Tx','Location','best')
axis equal
xlim([x1min x1max])
ylim([x2min x2max])
hold off


subplot(313)
hold on
histogram(w'*Xdog,15)
histogram(w'*Xcat,15)
legend('y_{dog}','y_{cat}')
hold off





