function [A,B,f] = generate_hash_codes8_matrix_update(S,m,n,q,A,B,option)
% Based on the TIP 2016 paper without the constraints
% Now to find A and B - how to do it ?
% the matrix version

% % % % % % Randomly initialize the matrices A and B
% % % % % a = -1; b = 1;
% % % % % A = (b-a)*rand(m,q,'double') + a; A = sign(A);
% % % % % B = (b-a)*rand(n,q,'double') + a; B = sign(B);

niter = 100;

f(1) = norm(S-(1/q)*sign(A)*sign(B).','fro');
fprintf('The original function value is :  f val is : %f \n', f(1));
tolerance = 1e-6;
for t=2:niter
    % update A
    temp = B.'*B;
    nu = 2*eigs(temp,1) + eps;
%     nu = 2*norm(temp,'fro') + eps;
    grad = 2*A*(B.'*B) - 2*q*S*B;
    A = A - grad*(1/nu);
    if option==1
        A(A<-1) = -1; A(A>1) = 1;
    elseif option==2
        A = sign(A);
    end
    
    % update B
    temp = A.'*A;
    nu = 2*eigs(temp,1) + eps;
%     nu = 2*norm(temp,'fro') + eps;
    grad = 2*B*(A.'*A) - 2*q*S.'*A;
    B = B - grad*(1/nu);
    if option==1
        B(B<-1) = -1; B(B>1) = 1;
    elseif option==2
        B = sign(B);
    end
    
    % Compute the function value
    f(t) = norm(S-(1/q)*sign(A)*sign(B).','fro');
    
    if mod(t,10)==0
        % Print the objective value
        fprintf('The iteration is : %i and f val is : %f \n', t, f(t));
    end
    
    if t>5
        if (f(t-1)-f(t))/f(t) <= tolerance
            break;
        end
    end
    
end

A = sign(A); B = sign(B);

