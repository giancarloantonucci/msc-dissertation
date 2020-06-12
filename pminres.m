function [u1, j] = pminres(M, A, u0, b, m, n)
% Preconditioned MINRES. The algorithm is taken from (Elman, Silvester and 
% Wathen, "Finite Elements And Fast Iterative Solvers With Applications In 
% Incompressible Fluid Dynamics" - p. 192).
%
% [U1, J] = PMINRES(M, A, U0, B, M, N) solves the block system of size M*N 
% with coefficient matrix A and known term B using MINRES with
% preconditioner M and initial guess U0. The outputs are the solution U1
% and the number of iterations until convergence J.
% 
% Giancarlo Antonino Antonucci, 2017.

mn = size(u0);
v0 = zeros(mn); w0 = zeros(mn); w1 = w0; g0 = 0;
v1 = b - A*u0;
z1 = fftsolve(M,v1,m,n); g1 = sqrt(z1'*v1);
eta = g1; s0 = 0; s1 = 0; c0 = 1; c1 = 1;
j = 1; r = v1;

while norm(r) > 1e-5 && j < 1000
    z1 = z1/g1;
    d = (A*z1)'*z1;
    if j == 1
        v2 = A*z1 - d/g1*v1;
    else
        v2 = A*z1 - d/g1*v1 - g1/g0*v0;
    end 
    z2 = fftsolve(M,v2,m,n);
    g2 = sqrt(z2'*v2);
    a0 = c1*d - c0*s1*g1;
    a1 = sqrt(a0^2 + g2^2);
    a2 = s1*d + c0*c1*g1;
    a3 = s0*g1;
    c2 = a0/a1; s2 = g2/a1;
    w2 = (z1 - a3*w0 - a2*w1)/a1;
    u1 = u0 + c2*eta*w2;
    eta = -s2*eta;
    
    z1 = z2;
    g0 = g1; g1 = g2;
    v0 = v1; v1 = v2;
    s0 = s1; s1 = s2;
    c0 = c1; c1 = c2;
    w0 = w1; w1 = w2;
    u0 = u1;
    j = j+1;
    
    r = b - A*u0;
end