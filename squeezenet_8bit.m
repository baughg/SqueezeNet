% Forward path implementation of SqueezeNet 
clc;
clear;
inputFile = '';     %Sample input image for which we want to extract 
                    %intermediate results. If intermed = 0, this value is
                    %not requried. 
meanFile = '';      %Path to the mean file of the dataset
cmp = 0;            %If cmp == 1, program checks the intermediate results
                    %with the expected results and print the difference. 
                    %If you want to use your own image file, set this to 0 
                    %in the confix.txt file.
global operation_id;
operation_id = 1;
Z_MAJOR = 0;
CHANNEL_MAJOR = 1;
fileId = fopen('Config.txt');
if (fileId == -1)
    error('Cannot find config.txt in the current directory');
end
line = fgets(fileId);
while (ischar(line)) 
    tokens = strsplit(line, '=');
    if (line(1) == '#')
        line = fgets(fileId);
        continue;
    end
    switch(strtrim(tokens{1}))
        case 'input_file'
            inputFile = strtrim(tokens{2});
        case 'mean_file'
            meanFile = strtrim(tokens{2});
        case 'cmp'
            cmp = str2double(strtrim(tokens{2}));
    end
    line = fgets(fileId);
end
fclose(fileId);
img = preproc(inputFile, meanFile);

figure(1);
image(uint8(img)); axis image; axis off; colormap(gray(256));
scale_factor = 127;
img = scale_and_quantise_max(img);
convolution_max = zeros(1,18);
% thres = [60471.5 31634.5 12599.75 20294.5 16296.25 13944.25 11778.25 26338.75 12000 16544.75 10269.5 16650.75 13664.5 15626 10621.5 23674.25 8922.5 5603];
thres = [72397 37066 11202 16939 14289 17973 10640 30756 13120 16579 12955 11371 13636 14743 13930 22782 11406 5303];

thres(:) = 128*128;
% lb2_thres(1:2) = thres(1:2);
thres(1) = bitshift(1,16);
thres(2) = bitshift(1,15);

lb2_thres = log10(thres) / log10(2);
lb2_thres = floor(lb2_thres);
lb2_thres = lb2_thres - 12;


%% Convolution Layer 1
%load('Intermed_Results\1_data.mat');
%img = data;
root_dir = 'test_data_8bit/';
tic
load('Params\conv1_w.mat');
load('Params\conv1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);

write_array_xyz([root_dir 'weight1_0_i8.bin'],weights,1,0 );
write_array_xyz([root_dir 'bias1_0_i8.bin'],bias,1,1 );

save_tensor([root_dir 'input0_0_i8.bin'],img);
% log_layer_io( layer_in,item_in, dir_io, io_type, append )
log_layer_io( 0,0,1, 3, 0, CHANNEL_MAJOR );
log_layer_io( 1,0,1, 1, 1, CHANNEL_MAJOR );
log_layer_io( 1,1,1, 2, 1, CHANNEL_MAJOR );
log_layer_io( 1,2,0, 3, 1, CHANNEL_MAJOR);
% layer_io_config( layer_in,item_in, scale, append )
layer_io_config(1,2,lb2_thres(1),0);
conv_rslt = conv(img, weights, bias, 7, 2, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(1) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(1));
save('conv_rslt_1_2.mat','conv_rslt');
load('Intermed_Results\2_conv1.mat');
if (cmp)
    fprintf('Max error in conv1: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Pooling Layer 1
pool_rslt = maxpool(conv_rslt, 3, 2);
load('Intermed_Results\3_pool1.mat');
if (cmp)
    fprintf('Max error in maxpool1: %f\n', max(abs(data(:) - pool_rslt(:))));
end
%% Fire Layer 2
load('Params\fire2_squeeze1x1_w.mat');
load('Params\fire2_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
write_array_zxy([root_dir 'weight2_0_i8.bin'],weights,2,0 );
write_array_xyz([root_dir 'bias2_0_i8.bin'],bias,2,1 );
conv_rslt = conv(pool_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(2) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(2));
load('Intermed_Results\4_fire2_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire2/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end
load('Params\fire2_expand1x1_w.mat');
load('Params\fire2_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
write_array_zxy([root_dir 'weight2_1_i8.bin'],weights,2,2 );
write_array_xyz([root_dir 'bias2_1_i8.bin'],bias,2,3 );
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire2_expand3x3_w.mat');
load('Params\fire2_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
write_array_zxy([root_dir 'weight2_2_i8.bin'],weights,2,4 );
write_array_xyz([root_dir 'bias2_2_i8.bin'],bias,2,5 );
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (55, 55, 128);
conv_rslt (:, :, 1:64) = conv_rslt_1;
conv_rslt (:, :, 65:128) = conv_rslt_2;
convolution_max(3) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(3));
load('Intermed_Results\9_fire2_concat.mat')
if (cmp)
    fprintf('Max error in Fire2: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Fire Layer 3
load('Params\fire3_squeeze1x1_w.mat');
load('Params\fire3_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
write_array_zxy([root_dir 'weight3_0_i8.bin'],weights,3,0 );
write_array_xyz([root_dir 'bias3_0_i8.bin'],bias,3,1 );
conv_rslt = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(4) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(4));
load('Intermed_Results\10_fire3_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire3/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end    
load('Params\fire3_expand1x1_w.mat');
load('Params\fire3_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
write_array_zxy([root_dir 'weight3_1_i8.bin'],weights,3,2 );
write_array_xyz([root_dir 'bias3_1_i8.bin'],bias,3,3 );
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire3_expand3x3_w.mat');
load('Params\fire3_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
write_array_zxy([root_dir 'weight3_2_i8.bin'],weights,3,4 );
write_array_xyz([root_dir 'bias3_2_i8.bin'],bias,3,5 );
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (55, 55, 128);
conv_rslt (:, :, 1:64) = conv_rslt_1;
conv_rslt (:, :, 65:128) = conv_rslt_2;
convolution_max(5) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(5));
load('Intermed_Results\15_fire3_concat.mat')
if (cmp)
    fprintf('Max error in Fire3: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Fire Layer 4
load('Params\fire4_squeeze1x1_w.mat');
load('Params\fire4_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(6) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(6));
load('Intermed_Results\16_fire4_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire4/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end    
load('Params\fire4_expand1x1_w.mat');
load('Params\fire4_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire4_expand3x3_w.mat');
load('Params\fire4_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (55, 55, 256);
conv_rslt (:, :, 1:128) = conv_rslt_1;
conv_rslt (:, :, 129:256) = conv_rslt_2;
convolution_max(7) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(7));
load('Intermed_Results\21_fire4_concat.mat')
if (cmp)
    fprintf('Max error in Fire4: %f\n', max(abs(data(:) - conv_rslt(:))));
end    

%% Pooling Layer 4
pool_rslt = maxpool(conv_rslt, 3, 2);
load('Intermed_Results\22_pool4.mat');
if (cmp)
    fprintf('Max error in maxpool4: %f\n', max(abs(data(:) - pool_rslt(:))));
end
%% Fire Layer 5
load('Params\fire5_squeeze1x1_w.mat');
load('Params\fire5_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(pool_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(8) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(8));
load('Intermed_Results\23_fire5_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire5/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end    
load('Params\fire5_expand1x1_w.mat');
load('Params\fire5_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire5_expand3x3_w.mat');
load('Params\fire5_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2); 
conv_rslt = zeros (27, 27, 256);
conv_rslt (:, :, 1:128) = conv_rslt_1;
conv_rslt (:, :, 129:256) = conv_rslt_2;
convolution_max(9) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(9));
load('Intermed_Results\28_fire5_concat.mat')
if (cmp)
    fprintf('Max error in Fire5: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Fire Layer 6
load('Params\fire6_squeeze1x1_w.mat');
load('Params\fire6_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(10) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(10));
load('Intermed_Results\29_fire6_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire6/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end    
load('Params\fire6_expand1x1_w.mat');
load('Params\fire6_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire6_expand3x3_w.mat');
load('Params\fire6_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (27, 27, 384);
conv_rslt (:, :, 1:192) = conv_rslt_1;
conv_rslt (:, :, 193:384) = conv_rslt_2;
convolution_max(11) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(11));
load('Intermed_Results\34_fire6_concat.mat')
if (cmp)
    fprintf('Max error in Fire6: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Fire Layer 7
load('Params\fire7_squeeze1x1_w.mat');
load('Params\fire7_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(12) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(12));
load('Intermed_Results\35_fire7_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire7/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end    
load('Params\fire7_expand1x1_w.mat');
load('Params\fire7_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire7_expand3x3_w.mat');
load('Params\fire7_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (27, 27, 384);
conv_rslt (:, :, 1:192) = conv_rslt_1;
conv_rslt (:, :, 193:384) = conv_rslt_2;
convolution_max(13) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(13));
load('Intermed_Results\40_fire7_concat.mat')
if (cmp)
    fprintf('Max error in Fire7: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Fire Layer 8
load('Params\fire8_squeeze1x1_w.mat');
load('Params\fire8_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(14) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(14));
load('Intermed_Results\41_fire8_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire8/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end
load('Params\fire8_expand1x1_w.mat');
load('Params\fire8_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire8_expand3x3_w.mat');
load('Params\fire8_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (27, 27, 512);
conv_rslt (:, :, 1:256) = conv_rslt_1;
conv_rslt (:, :, 257:512) = conv_rslt_2;
convolution_max(15) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(15));
load('Intermed_Results\46_fire8_concat.mat')
if (cmp)
    fprintf('Max error in Fire8: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Pooling Layer 8
pool_rslt = maxpool(conv_rslt, 3, 2);
load('Intermed_Results\47_pool8.mat');
if (cmp)
    fprintf('Max error in maxpool8: %f\n', max(abs(data(:) - pool_rslt(:))));
end
%% Fire Layer 9
load('Params\fire9_squeeze1x1_w.mat');
load('Params\fire9_squeeze1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(pool_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt = relu(conv_rslt);
convolution_max(16) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(16));
load('Intermed_Results\48_fire9_squeeze1x1.mat');
if (cmp)
    fprintf('Max error in Fire9/SQ1: %f\n', max(abs(data(:) - conv_rslt(:))));
end
load('Params\fire9_expand1x1_w.mat');
load('Params\fire9_expand1x1_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_1 = conv(conv_rslt, weights, bias, 1, 1, 0, 1);
conv_rslt_1 = relu(conv_rslt_1);
load('Params\fire9_expand3x3_w.mat');
load('Params\fire9_expand3x3_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt_2 = conv(conv_rslt, weights, bias, 3, 1, 1, 1);
conv_rslt_2 = relu(conv_rslt_2);
conv_rslt = zeros (13, 13, 512);
conv_rslt (:, :, 1:256) = conv_rslt_1;
conv_rslt (:, :, 257:512) = conv_rslt_2;
convolution_max(17) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(17));
load('Intermed_Results\53_fire9_concat.mat')
if (cmp)
    fprintf('Max error in Fire9: %f\n', max(abs(data(:) - conv_rslt(:))));
end

%% Convolution Layer 10
load('Params\conv10_w.mat');
load('Params\conv10_b.mat');
weights = quantise_array(weights,scale_factor);
bias = quantise_array(bias,scale_factor);
conv_rslt = conv(conv_rslt, weights, bias, 1, 1, 1, 1);
conv_rslt = relu(conv_rslt);
convolution_max(18) = max(conv_rslt(:));
conv_rslt = scale_and_quantise_var(conv_rslt, lb2_thres(18));
load('Intermed_Results\54_conv10.mat');
if (cmp)
    fprintf('Max error in conv10: %f\n', max(abs(data(:) - conv_rslt(:))));
end
%% Average Pooling Layer
pool_rslt = avgpool(conv_rslt, 15, 1);
load('Intermed_Results\55_pool10.mat');
if (cmp)
    fprintf('Max error in avgpool10: %f\n', max(abs(data(:) - pool_rslt(:))));
end
%% Softmax
soft_rslt = softmax(pool_rslt);
a = squeeze(soft_rslt);
class = find(max(a) == a) - 1
load('Intermed_Results\56_prob.mat');
if (cmp)
    fprintf('Max error in softmax: %f\n', max(abs(data(:) - soft_rslt(:))));
end
toc