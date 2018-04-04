function log_layer_io( layer_in,item_in, dir_io, io_type, append, ORDER )
global operation_id;

if append == 0
    fid = fopen(['operation/io' num2str(operation_id) '.bin'],'wb');
else
    fid = fopen(['operation/io' num2str(operation_id) '.bin'],'ab');
end;

fwrite(fid,layer_in,'uint16');
fwrite(fid,item_in,'uint16');
fwrite(fid,dir_io,'uint16');
fwrite(fid,io_type,'uint16');
fwrite(fid,ORDER,'uint32');
fclose(fid);
end

