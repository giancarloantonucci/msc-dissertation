%% Preconditioned MINRES with absolute value circulant preconditioner for 
% the scalar heat equation with convective source term.
% 
% Giancarlo Antonino Antonucci, 2017.

%% Grid
n = 100;            % space grid points
x0 = 0;            	% space start
xN = 1;            	% space end
dx = (xN-x0)/(n+1);	% space step size
x = x0:dx:xN;      	% space grid

m = 100;         	% time grid points
t0 = 0;             % time start
tM = 1;             % time end
dt = (tM-t0)/m;     % time step size
t = t0:dt:tM;       % time grid

%% Parameters
mu = dt/dx^2;       % grid ratio
beta = 10;          % source parameter

%% Source Term and Conditions
u0 = exp(x(2:n+1)').*sin(pi*x(2:n+1)');

left = 0*ones(1,m+1); right = sin(pi*t(1:m+1));
g = [left; zeros(n-2,m+1); right]; g = g(:);

p = [left; zeros(n-2,m+1); -right]; p = p(:);

%% Linear system
T = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)], [-1 0 1], n, n);
S = spdiags([-ones(n,1) ones(n,1)], [-1 1], n, n);
A0 = speye(n) + mu*T - .5*dt/dx*beta*S;
A1 = -speye(n);
A = kron(speye(m),A0) + kron(spdiags(ones(m,1), -1, m, m),A1);
G = kron(speye(m),A0) + kron(spdiags(exp(2i*pi*(0:m-1)'/m), 0, m, m),A1);

b = mu*g(n+1:end) + .5*dt/dx*beta*p(n+1:end);
b(1:n) = b(1:n) + u0;

%% preconditioned MINRES with absolute value circulant preconditioner
for i = 1:m
    idx = (i-1)*n+1:i*n;
    G(idx,idx) = sparse(full(G(idx,idx)'*G(idx,idx))^(1/2));
end

A = A(end:-1:1,:); % A = Y*A; 
b = b(end:-1:1); % b = Y*b;
[u, j] = pminres(G, A, zeros(n*m,1), b, m, n);

%% Plot
result = zeros(n+2,m+1);
result(1,:) = left;
result(end,:) = right;
result(2:n+1,1) = u0;
result(2:n+1,2:m+1) = reshape(real(u),n,m);

mesh(t,x,result)
xlabel('Time $t$'), ylabel('Space $x$'), zlabel('Solution $u$')
title(['iterations required = ' num2str(j)])
view([45,25])