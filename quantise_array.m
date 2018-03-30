function Q = quantise_array( a, scale_factor )

a = a * scale_factor;
Q = int8(a);
Q = double(Q);
end

