function vec2 = fftsolve(M, vec1, m, n)
% Highly parallelisable solver. This is intended as a component of PMINRES.
%
% VEC2 = FFTSOLVE(M, VEC1, M, N) solves the M*N system M*VEC2 = VEC1 using
% the fast fourier transform and extracting a block diagonal from the 
% circulant in time matrix M and inverting its blocks. Look at the main
% text for further information.
% 
% Note that
%
% M = kron(U,eye(n))*X(:)
%   = X*(U.') and M(:)
%   = fft(X.').'/sqrt(m) and M2(:)
% 
% are all equivalent, but the last one is the least expensive.
%
% For 1-level use instead: vec2 = ifft(bsxfun(@times,M,fft(vec1))).
% 
% Giancarlo Antonino Antonucci, 2017.

vec2 = zeros(m*n,1);
for i = 1:n
    vec2(i:n:m*n,1) = ifft(vec1(i:n:m*n));
end
for i = 1:m
    idx = (i-1)*n+1:i*n;
    vec1(idx) = M(idx,idx)\vec2(idx);
end
for i = 1:n
    vec2(i:n:m*n,1) = fft(vec1(i:n:m*n));
end