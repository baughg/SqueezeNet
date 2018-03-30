function Q = scale_and_quantise_max( a )
max_a = max(abs(a(:)));

scale_factor = 127.0 / max_a;

a = a * scale_factor;
Q = int8(a);
Q = double(Q);
end

