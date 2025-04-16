%{
Block Lanczos algorithm example.
A demo code to verify the algorithm and for the further C code develpoment.
%}


%%
%Parameters:
n = 6;          %Size of A matrix
b = 2;          %block size of nu matrix
iter = n/b;     %iterations
alpha = zeros(b,b,iter);
beta = zeros(b,b,iter);

%init A matrix, a hand put in example
A = zeros(n);
A = [2 1 0 0 0 1;...
     1 2 1 0 0 0;...
     0 1 2 1 0 0;...
     0 0 1 2 1 0;...
     0 0 0 1 2 1;...
     1 0 0 0 1 2];

%init B matrix, block-vector matrix
B = zeros(n);

%hand put in the first block
%!Question!"
%"Rank Deficiency in Residual" / "Incomplete Subspace Expansion" /
%"Breakdown of the Process" is much important here.
%'not “adjacent” in the standard basis' is a common choice.
%I see this issue but not much knowledge about it. 

%I just hand put in a working example here and a systematic algorithm to
%initialize the first block is future work.
B(:,1) = [1 0 0 0 0 0];
B(:,2) = [0 1 0 0 0 0];


% B=randn(6,2);
% Deflation


% options for starting vectors.
% 
%



%%
%First loop
[B,r] = qr(B,"econ","vector");

W = A*B(:,1:b);
alpha(:,:,1) = B(:,1:b)' * W;
W = W - B(:,1:b)*alpha(:,:,1);


%%
%Block Lanczos loop
for i = 2:iter

    %options for orthogonaliztoin W: Qr, sqroot of grammian,  diagonal
    %scaling 


% diagonal scaling
% PhD thesis of Kathrin Lund
    r=diag(sqrt(sum(conj(W).*W,1)));
    q=W/r;


% orthogonaliztion by using matrix square roots (symmetric)
%     M=W'*W;
%     r=sqrtm(M);
%     q=W/r;

     [q,r] = qr(W,"econ","vector");
%     %!Question!"
%     %It is annoying that Matlab usually gives a negative or arbitrarily Q and
%     %R, which is correct but contradict to the analytic calculation.
%     %The best reason I could find is it is due to the fundamental algorithm Matlab chooses.
%     %I cannot fix this issue perfectly but only a workaround.
%     %I implement it so I can have a identical result to my analytic calculation.
     D = diag(sign(diag(r)));
     q = q*D;
     r = D*r;

    B(:,(i-1)*b+1:i*b) = q;
    beta(:,:,i) = r;
    W = A * B(:,(i-1)*b+1:i*b) - B(:,(i-2)*b+1:(i-1)*b) * beta(:,:,i)';
    alpha(:,:,i) = B(:,(i-1)*b+1:i*b)' * W;
    W = W - B(:,(i-1)*b+1:i*b)*alpha(:,:,i);

end
%note: not identical to the analytical calculation, in progress.


%%
%assable tridiagonal matrix

ind =@(x) (1:b)+(x-1)*b;
T=zeros(b*iter,b*iter);
for i=1:iter-1
    T(ind(i),ind(i))=alpha(:,:,i);
    T(ind(i+1),ind(i))=beta(:,:,i+1);
    T(ind(i),ind(i+1))=beta(:,:,i+1)';
end
T(ind(iter),ind(iter))=alpha(:,:,iter);

disp('error in runnign lanczos')
%corretness check
norm((B'*A*B - T),'f')/norm(T)

%{
"correctness check", "break down" are not implemented in this demo yet.
%}