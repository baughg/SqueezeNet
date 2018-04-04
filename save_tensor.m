function save_tensor( filename, T )
T = int8(T);

% [C, W, H] = size(T);
fid = fopen(filename, 'wb');
layer = 0;
item = 0;

[W,H,Cn,Cp] = size(T);
order = 3;
fwrite(fid,layer,'uint16');
fwrite(fid,item,'uint16');
fwrite(fid,order,'int32');
fwrite(fid,W,'int32');
fwrite(fid,H,'int32');
fwrite(fid,Cn,'int32');
fwrite(fid,Cp,'int32');



for c = 1:Cn
    chan = T(:,:,c)';    
    fwrite(fid,chan(:),'int8');
end

fclose(fid);
end

