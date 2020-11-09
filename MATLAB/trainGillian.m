rng(1);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');

result_dir='/isilon/datalake/cialab/scratch/cialab/tet/python/AttentionDeepMIL/heatmap/results/*.png.tif';
slide_dir='/isilon/datalake/cialab/original/cialab/image_database/d00127/Scans/';
a=miceReadMap('slideMap_fixed.xlsx');

results=dir(result_dir);
openslide_load_library;

res=[64,323,30,452,412,462,401,270,436,245,169,416,459,196,352,325,415,21];
sus=[90,37,372,111,233,3,214,215];
ss=[130,231,86,279,286,274,379,150,55,307,129,31,15,78,178,125,155,299,40,293,135,365,69,313,207,133,356,210,87,156];

ri=randperm(length(results));

sf=1;
lev=0;
parfor i=1:length(ri)
    n=results(ri(i)).name;
    n=strsplit(n,'.');
    n=str2num(n{1});
    for j=1:size(a,1)
        if find(a.N{j}==n)
            break;
        end
    end
    
    if find(res==n) k='nss'; end
    if find(sus==n) k='nss'; end
    if find(ss==n)  k='ss'; end
    
    jj=j;
    mask=imread(strcat('./mouseMasks/',num2str(n),'.png'));
    p=regionprops(mask,'BoundingBox');
    for j=1:length(p)
    %j=randi([1,length(p)]);
        r=p(j).BoundingBox;
        imf=imfinfo(fullfile(results(ri(i)).folder,results(ri(i)).name));
        w=imf.Width;
        h=imf.Height;
        scale=((w/size(mask,2))+(h/size(mask,1)))/2;
        r=int64(r.*scale);
        
        if 2^30>int64(r(3)/sf)*int64(r(4)/sf)
            fprintf('using %i,%i\n',ri(i),j);
            heatmap=imread(fullfile(results(ri(i)).folder,results(ri(i)).name),'PixelRegion',{[r(2),r(2)+r(4)],[r(1),r(1)+r(3)]});
            heatmap=imresize(heatmap,1/sf);
            f=openslide_open(fullfile(slide_dir,a.Name{jj}));
            %im=openslide_read_region(f,r(1),r(2),r(3),r(4));
            im=openslide_read_region(f,int64(r(1)/sf),int64(r(2)/sf),int64(r(3)/sf),int64(r(4)/sf),'level',lev);

            imwrite(heatmap,strcat('trainGillian/',k,'/hm_',num2str(ri(i)),'_',num2str(j),'.jp2'),'Mode','lossless');
            imwrite(im(:,:,2:4),strcat('trainGillian/',k,'/he_',num2str(ri(i)),'_',num2str(j),'.jp2'),'Mode','lossless');
        else
            fprintf('skipping %i,%i\n',ri(i),j);
        end
    end
end
    