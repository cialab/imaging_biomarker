rng(1);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');

result_dir='/isilon/datalake/cialab/scratch/cialab/tet/python/AttentionDeepMIL/heatmap/results_all/';
slide_dir='/isilon/datalake/cialab/original/cialab/image_database/d00127/Scans/';
a=miceReadMap('slideMap_fixed.xlsx');

openslide_load_library;

SSasR=[31,59,61,66,82,109,121,132,161,162,167,178,263,282,290];
SSasS=[7,12,26,28,51,70,74,93,94,97,117,124,130,156,163,164,166,175,240,261,274,279,281,295,313,350];
SasSS=[6,19,22,45,56,126,129,209,407];
RasSS=[38,46,68,115,254,255,284,318,366,466];

folders=cat(1,repmat("SSasR",length(SSasR),1),...
    repmat("SSasS",length(SSasS),1),...
    repmat("SasSS",length(SasSS),1),...
    repmat("RasSS",length(RasSS),1));
slides=cat(2,SSasR,SSasS,SasSS,RasSS)';

sf=1;
lev=0;
parfor i=1:length(slides)
    try
        n=slides(i);
        k=folders(i);
        for j=1:size(a,1)
            if find(a.N{j}==n)
                break;
            end
        end
        jj=j;
        mask=imread(strcat('./mouseMasks/',num2str(n),'.png'));
        p=regionprops(mask,'BoundingBox');
        for j=1:length(p)
        %j=randi([1,length(p)]);
            r=p(j).BoundingBox;
            imf=imfinfo(strcat(result_dir,num2str(n),'.png.tif'));
            w=imf.Width;
            h=imf.Height;
            scale=((w/size(mask,2))+(h/size(mask,1)))/2;
            r=int64(r.*scale);

            if 2^30>int64(r(3)/sf)*int64(r(4)/sf)
                fprintf('using %i,%i\n',n,j);
                heatmap=imread(strcat(result_dir,num2str(n),'.png.tif'),'PixelRegion',{[r(2),r(2)+r(4)],[r(1),r(1)+r(3)]});
                heatmap=imresize(heatmap,1/sf);
                f=openslide_open(fullfile(slide_dir,a.Name{jj}));
                %im=openslide_read_region(f,r(1),r(2),r(3),r(4));
                im=openslide_read_region(f,int64(r(1)/sf),int64(r(2)/sf),int64(r(3)/sf),int64(r(4)/sf),'level',lev);

                imwrite(heatmap,strcat('trestGillian/',k,'/hm_',num2str(n),'_',num2str(j),'.jp2'),'Mode','lossless');
                imwrite(im(:,:,2:4),strcat('trestGillian/',k,'/he_',num2str(n),'_',num2str(j),'.jp2'),'Mode','lossless');
            else
                fprintf('skipping %i,%i\n',n,j);
            end
        end
    catch
        fprintf('Caught error on %i\n',n);
    end
end