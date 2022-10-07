%% Generate data

clearvars, close all

% class 1
% dog
dn = 100;
dm = [1,1.5];
ds = 0.2;
rho = 0;
Xdog = mvnrnd(dm,ds^2*[1,rho;rho,1],dn)';
Xdog = [ones(1,dn); Xdog];
Tdog = [1;0]*ones(1,dn);

% class 2
% cat
cn = 80;
cm = [2,1];
cs = 0.2;
rho = 0.0;
Xcat = mvnrnd(cm,cs^2*[1,rho;rho,1],cn)';
Xcat = [ones(1,cn); Xcat];
Tcat = [0;1]*ones(1,cn);

% join data
T = [Tdog Tcat];
X = [Xdog Xcat];

% Plot data
figure(1), clf, hold on,
plot(Xdog(2,:),Xdog(3,:),'+','MarkerSize',5)
plot(Xcat(2,:),Xcat(3,:),'o','MarkerSize',5)
legend('Dog','Cat')
hold off

%% Estimate W

W = pinv(X')*T';

%% Make vizualization grid

ngrid = 100;
Xgrid = zeros(3,ngrid^2);
x1grid = linspace(min(X(2,:)),max(X(2,:)),ngrid);
x2grid = linspace(min(X(3,:)),max(X(3,:)),ngrid);

nind = 1;
for n1 = 1:ngrid
    for n2 = 1:ngrid
        Xgrid(:,nind) = [1,x1grid(n1),x2grid(n2)]';
        nind = nind + 1;
    end
end

%% Calculate predictions on grid

Ypred = W'*Xgrid;

y1 = Ypred(1,:)';
y2 = Ypred(2,:)';

y1mat = reshape(y1,ngrid,ngrid);
y2mat = reshape(y2,ngrid,ngrid);


%% Plot results

% Per classifier
figure(2), clf, 
subplot(121), hold on,
contourf(x1grid,x2grid,y1mat,50)
plot(Xdog(2,:),Xdog(3,:),'k+','MarkerSize',5)
plot(Xcat(2,:),Xcat(3,:),'ko','MarkerSize',5)
legend('y - dog','Dog','Cat','Location','best')
hold off

subplot(122), hold on,
contourf(x1grid,x2grid,y2mat,50)
plot(Xdog(2,:),Xdog(3,:),'k+','MarkerSize',5)
plot(Xcat(2,:),Xcat(3,:),'ko','MarkerSize',5)
legend('y - cat','Dog','Cat','Location','best')
hold off


% Decision boundaries, max of classifiers
[c,maxind] = max(Ypred,[],1);
cmat = reshape(c,ngrid,ngrid);
classmat = reshape(maxind,ngrid,ngrid);

figure(3), clf
subplot(121), hold on,
contourf(x1grid,x2grid,cmat,50)
plot(Xdog(2,:),Xdog(3,:),'k+','MarkerSize',5)
plot(Xcat(2,:),Xcat(3,:),'ko','MarkerSize',5)
legend('max(y)','Dog','Cat','Location','best')
hold off

subplot(122), hold on,
contour(x1grid,x2grid,classmat,2)
plot(Xdog(2,:),Xdog(3,:),'+','MarkerSize',5)
plot(Xcat(2,:),Xcat(3,:),'o','MarkerSize',5)
legend('Decision boundary','Dog','Cat','Location','best')
hold off

