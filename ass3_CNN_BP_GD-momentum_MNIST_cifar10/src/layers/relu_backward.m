function dldx = relu_backward(x, dldy)
    %error('Implement this!');
    x(x>0) = 1;
    x(x<=0) = 0;
    dldx = x .* dldy;
        
    
end
