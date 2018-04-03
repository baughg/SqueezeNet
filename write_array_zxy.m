function write_array_zxy(filename,Wght ,layer, item)
fid = fopen(filename,'wb');


[W,H,Cn,Cp] = size(Wght);
order = 2;
fwrite(fid,layer,'uint16');
fwrite(fid,item,'uint16');
fwrite(fid,order,'int32');
fwrite(fid,W,'int32');
fwrite(fid,H,'int32');
fwrite(fid,Cp,'int32');
fwrite(fid,Cn,'int32');
wght_rf = zeros(W,H,Cp,Cn);

% for s = 1:Cn
%     for cs = 1:Cp
%         wght_rf(:,:,cs,s) = Wght(:,:,s,cs);
%     end
% end
% 
% Wght = wght_rf;
% clear wght_rf;


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



