function Q = scale_and_quantise_var( a, shft )
% scale_factor = 127 / shft;
% a = a * scale_factor;

a = bitshift(a,-shft);
a = bitshift(a,-5);

Q = int8(a);
Q = double(Q);
end

