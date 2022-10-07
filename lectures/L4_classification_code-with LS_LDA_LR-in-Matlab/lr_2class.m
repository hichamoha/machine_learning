%% Generate data

clearvars, close all

% class 1
% dog
dn = 100;
dm = [1,1.5];
ds = 0.4;
rho = 0;
Xdog = mvnrnd(dm,ds^2*[1,rho;rho,1],dn)'
Tdog = repmat({'Dog'},1,dn)';

% class 2
% cat
cn = 80;
cm = [2,1];
cs = 0.4;
rho = 0.0;
Xcat = mvnrnd(cm,cs^2*[1,rho;rho,1],cn)'
Tcat = repmat({'Cat'},1,cn)';
 

% join data
T = [Tdog; Tcat];
X = [Xdog Xcat];

% Plot data
figure(1), clf, hold on,
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
legend('Dog','Cat','Hippo')
hold off

%% Estimate W

W = mnrfit(X',categorical(T));


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

y1 = (W'*Xgrid);
p1 = 1./(1+exp(y1));
p2 = 1-p1;


pcat = 1./(1+ exp(W'*[ones(1,dn); Xdog]));
pdog = 1-pcat;

y1mat = reshape(y1,ngrid,ngrid);
p1mat = reshape(p1,ngrid,ngrid);


%% Plot results

figure(2), clf, 
subplot(121), hold on,
contour(x1grid,x2grid,p1mat,1)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
legend('Probabilities','Dog','Cat','Location','best')
hold off

subplot(122), hold on,
contour(x1grid,x2grid,y1mat,50)
plot(Xdog(1,:),Xdog(2,:),'+','MarkerSize',5)
plot(Xcat(1,:),Xcat(2,:),'o','MarkerSize',5)
legend('Logits','Dog','Cat','Location','best')
hold off

