function Copy_of_makeBags(b)
rng(1);
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

% Configuration
load('/isilon/datalake/cialab/scratch/cialab/tet/MATLAB/mil/ss_labels.mat');
slidedir='/isilon/datalake/cialab/original/cialab/image_database/d00127/Scans';
fg=0.5;
wd='/isilon/datalake/cialab/scratch/cialab/tet/MATLAB/mil/datasets';
patch_sizes=[16];%[32,64,128,256];
meanstds=[10000;2000];%[10,50,100,500,1000,5000;
         %  2,10,20,100,200,1000];
m=miceReadMap('slideMap_fixed.xlsx');

% Loop
m=m(randperm(size(m,1)),:);
l=round(linspace(0,size(m,1),11));
for n=l(b)+1:l(b+1)
    slide=m.Name(n);
    mice=m.N{n};
    
    % Open file
    fp=openslide_open(fullfile(slidedir,char(slide)));
    [y,x]=openslide_get_level0_dimensions(fp);
    
    for i=1:length(mice)        
        mask=imread(fullfile('./mouseMasks',strcat(num2str(mice(i)),'.png')));
        fprintf('%s: %i\n',char(slide),mice(i));
        tic;
        [idx,val]=find(mouseNum==mice(i));
        for ps=patch_sizes
            for meanstd=meanstds
                makeBag(fp,y,x,mask,...
                    mice(i),labels(idx),...
                    round(normrnd(meanstd(1),meanstd(2))),ps,fg,...
                    fullfile(wd,num2str(ps),...
                    strcat(num2str(meanstd(1)),'_',num2str(meanstd(2)))));
            end
        end
        toc;
    end
    
    openslide_close(fp);
end

end