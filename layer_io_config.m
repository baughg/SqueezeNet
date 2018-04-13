function layer_io_config( layer_in,item_in, scale, append )
global operation_id;

if append == 0
    fid = fopen(['operation/io_cfg' num2str(operation_id) '.bin'],'wb');
else
    fid = fopen(['operation/io_cfg' num2str(operation_id) '.bin'],'ab');
end;

fwrite(fid,layer_in,'uint16');
fwrite(fid,item_in,'uint16');
fwrite(fid,scale,'int32');

fclose(fid);
end

