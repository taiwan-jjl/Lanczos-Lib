%{
Basic Lanczos algorithm example.
A demo code to verify the algorithm and for the further C code develpoment.
%}

%%
% Construct the input square matrix A(n,n) from its known eigenvalues and
% eigenvectors.

evec1 = [1 0 0];
evec2 = [0 1 0];
evec3 = [0 0 1];
eval  = [2 3 5];
eval_mat = diag(eval);
evec_mat = transpose([normalize(evec1,"norm",2); normalize(evec2,"norm",2); normalize(evec3,"norm",2)]);
%A = evec_mat * eval_mat * inv(evec_mat);
A = evec_mat * eval_mat / evec_mat;

%%
% Verify the A matrix is constructed correctly.

[V, D] = eig(A); % For this trivial example, check "V = evec_mat" and "D = eval_mat" by eyes. 

%%
% Use the matrix A to demo the Lanczos algorithm.
% Set beta_0 = beta(1) = 0, 
%     omega_0 = omega(1) = 0, 
%     nu_1 = nu(1) = transpose([1 0 0]) for simplicity.

A = [6 2 0; 2 3 1; 0 1 4];
% For simplicity, a general A matrix, which is constructed above by any
% eigenvalues and eigenvectors, is not demoed here.
% Use a manually designed A matrix instead.

iter = 3;
alpha = zeros(iter,1);
beta = zeros(iter+1,1);
omega = zeros(iter, iter);
nu = zeros(iter, iter+2);
nu(:,2) = transpose([1 0 0]);
for i = 1:iter
    omega(:,i) = A*nu(:,i+1) - beta(i)*nu(:,i);
    alpha(i) = transpose(nu(:,i+1)) * omega(:,i);
    omega(:,i) = omega(:,i) - alpha(i)*nu(:,i+1);
    beta(i+1) = norm(omega(:,i),2);
    nu(:,i+2) = omega(:,i) / beta(i+1);
end

%%
% T matrix

T = diag(alpha)+diag(beta(2:3),1)+diag(beta(2:3),-1);
A == T

%{
"correctness check", "break down" are not implemented in this demo yet.
%}