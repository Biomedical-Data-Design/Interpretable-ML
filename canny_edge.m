clear all;
close all;


I = imread('0a747cb3-c720-4572-a661-ab5670a5c42e.png');

imshow(I);

I_gray = rgb2gray(I);
I_hc = imadjust(I_gray);

[~,threshold] = edge(I_hc,'sobel');
fudgeFactor = 0.5;
BWs = edge(I_hc,'sobel',threshold * fudgeFactor);


imshow(BWs);

se90 = strel('line',3,75);
se0 = strel('line',3,0);

BWsdil = imdilate(BWs,[se90 se0]);

imshow(BWsdil)

BWdfill = imfill(BWsdil, 'holes');

imshow(BWdfill)

BWnobord = imclearborder(BWdfill,1);

imshow(BWnobord)

seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
f=figure;
imshow(BWfinal)

%% Second pass

[~,threshold] = edge(I_hc,'sobel');
fudgeFactor = 0.5;
BWsv2 = edge(BWfinal,'sobel',threshold * fudgeFactor);
f=figure;
imshow(BWsv2)
BWsdil2 = imdilate(BWsv2,[se90 se0]);

imshow(BWsdil2)

BWdfill2 = imfill(BWsdil2, 'holes');

imshow(BWdfill2)

BWnobord2 = imclearborder(BWdfill2,1);

imshow(BWnobord2)
% 
% seD = strel('diamond',1);
% BWfinal2 = imerode(BWnobord2,seD);
% BWfinal2 = imerode(BWfinal2,seD);
% f=figure;
% imshow(BWfinal2)

s = size(I);

for i = 1:s(1)
    for j = 2:s(2)
        if BWfinal(i,j) == 0
            I_gray(i,j) = 0;
        end
    end
end
f = figure;
imshow(I_gray)

