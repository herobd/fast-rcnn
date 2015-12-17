
function  do_selective_search_for_im(im_file,save_file)

addpath '/home/brianld/fast-rcnn/lib/datasets/SelectiveSearch/'
addpath '/home/brianld/fast-rcnn/lib/datasets/SelectiveSearch/Dependencies'
addpath '/home/brianld/fast-rcnn/lib/datasets/SelectiveSearch/Dependencies/FelzenSegment'

im = imread(im_file);
if size(im,3)==1
   im = cat(3,im,im,im);
end
boxes = selective_search_boxes(im);

save(save_file,'boxes');
end
