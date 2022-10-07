%% Generate data
clearvars, close all

% class 1
% dog
dn = 100;
dm = [1,2];
ds = 0.2;
rho = 0.7;
%Xdog = ds*randn(2,dn) + dm(:)*ones(1,dn);
Xdog = mvnrnd(dm,ds^2*[1,rho;rho,1],dn)'
Tdog = repmat({'Dog'},1,dn);

% class 2
% cat
cn = 80;
cm = [2,1];
cs = 0.2;
rho = 0.7;
%Xcat = cs*randn(2,cn) + cm(:)*ones(1,cn);
Xcat = mvnrnd(cm,cs^2*[1,rho;rho,1],cn)'
Tcat = repmat({'Cat'},1,cn);

% class 3
% hippo
hn = 120;
hm = [1.5,1.5];
hs = 0.2;
rho = 0.7;
Xhippo = mvnrnd(hm,hs^2*[1,rho;rho,1],hn)';
Thippo = repmat({'Hippo'},1,hn);

% join data
T = [Tdog Tcat Thippo];
X = [Xdog Xcat Xhippo];

% Plot data
figure(1), clf, hold on,
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
plot(Xhippo(1,:),Xhippo(2,:),'*','MarkerSize',5)
legend('Dog','Cat','Hippo')

hold off

%% Estimate W

W = mnrfit(X',categorical(T'))

%% Make vizualization grid

ngrid = 100;
Xgrid = zeros(3,ngrid^2);
x1grid = linspace(min(X(1,:)),max(X(1,:)),ngrid);
x2grid = linspace(min(X(2,:)),max(X(2,:)),ngrid);

nind = 1;
for n1 = 1:ngrid
    for n2 = 1:ngrid
        Xgrid(:,nind) = [1,x1grid(n1),x2grid(n2)]';
        nind = nind + 1;
    end
end

%% Calculate probabilities

P = mnrval(W,Xgrid(2:3,:)');

pcat = P(:,1);
pdog = P(:,2);
phippo = P(:,3);

pcatmat = reshape(pcat,ngrid,ngrid);
pdogmat = reshape(pdog,ngrid,ngrid);
phippomat = reshape(phippo,ngrid,ngrid);

%% Plot results

% One vs all boundaries
figure(2), clf, 
subplot(131), hold on,
contour(x1grid,x2grid,pcatmat,50)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
plot(Xhippo(1,:),Xhippo(2,:),'*','MarkerSize',5)
legend('P(cat)','Dog','Cat','Hippo','Location','best')

hold off

subplot(132), hold on,
contour(x1grid,x2grid,pdogmat,50)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
plot(Xhippo(1,:),Xhippo(2,:),'*','MarkerSize',5)
legend('P(dog)','Dog','Cat','Hippo','Location','best')
hold off


subplot(133), hold on,
contour(x1grid,x2grid,phippomat,50)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
plot(Xhippo(1,:),Xhippo(2,:),'*','MarkerSize',5)
legend('P(hippo)','Dog','Cat','Hippo','Location','best')
hold off


% Decision boundaries
[c,maxind] = max(P,[],2);
classmat = reshape(maxind,ngrid,ngrid);

figure(5), clf, hold on,
contour(x1grid,x2grid,classmat,2,'k')
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
plot(Xhippo(1,:),Xhippo(2,:),'*','MarkerSize',5)
legend('Decision boundaries','Dog','Cat','Hippo','Location','best')
hold off

