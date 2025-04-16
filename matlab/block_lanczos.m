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
A = [2 1 0 0 0 0;...
     1 2 1 0 0 0;...
     0 1 2 1 0 0;...
     0 0 1 2 1 0;...
     0 0 0 1 2 1;...
     0 0 0 0 1 2];

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
B(:,2) = [0 0 1 0 0 0];


%%
%First loop
W = A*B(:,1:b);
alpha(:,:,1) = B(:,1:b)' * W;
W = W - B(:,1:b)*alpha(:,:,1);


%%
%Block Lanczos loop
for i = 2:iter

    [q,r] = qr(W,"econ","vector");
    %!Question!"
    %It is annoying that Matlab usually gives a negative or arbitrarily Q and
    %R, which is correct but contradict to the analytical calculation.
    %The best reason I could find is it is due to the fundamental algorithm Matlab chooses.
    %I cannot fix this issue perfectly but only a workaround.
    %I implement it so I can have an identical result to my analytical calculation.
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



    
%{
"correctness check", "break down" are not implemented in this demo yet.
%}