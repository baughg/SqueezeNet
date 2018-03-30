function write_array_xyz(filename,Wght,layer, item )
fid = fopen(filename,'wb');

[W,H,Cn,Cp] = size(Wght);
order = 1;
fwrite(fid,layer,'uint16');
fwrite(fid,item,'uint16');
fwrite(fid,order,'int32');

fwrite(fid,W,'int32');
fwrite(fid,H,'int32');
fwrite(fid,Cn,'int32');
fwrite(fid,Cp,'int32');

for co = 1:Cp
    weight_channel = zeros(W,H);
    for ci = 1:Cn
        weight_channel = Wght(:,:,ci,co);
        weight_channel = weight_channel';
        fwrite(fid,weight_channel(:),'int8');
    end
end

fclose(fid);
end



