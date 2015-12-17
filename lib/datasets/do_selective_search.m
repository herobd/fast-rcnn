
function  do_selective_search(im_indexes_file,im_loc,im_ext,save_file)

addpath '/home/brianld/fast-rcnn/lib/datasets/SelectiveSearch/'
addpath '/home/brianld/fast-rcnn/lib/datasets/SelectiveSearch/Dependencies'
addpath '/home/brianld/fast-rcnn/lib/datasets/SelectiveSearch/Dependencies/FelzenSegment'
addpath '/home/brianld/fast-rcnn/'

boxes={};
images={};
[im_indexes] = ...
        textread(im_indexes_file,'%s');
for i = 1:numel(im_indexes)
    im_path = fullfile(im_loc,[im_indexes{i} im_ext]);
    im = imread(im_path);
    if size(im,3)==1
       im = cat(3,im,im,im);
    end
    %im=im2double(im);
    im_boxes = selective_search_boxes(im);
    [height, width, toss] = size(im);
    for ii=1:size(im_boxes,1)
        box=im_boxes(ii,:);
        assert(box(3)<=height && box(4)<=width);
        assert(box(1)>0 && box(2)>0);
    end
    boxes(end+1)={im_boxes};
    images(end+1)={im_indexes{i}};
end

save(save_file,'boxes','images');
end
