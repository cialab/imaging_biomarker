result_dir='/isilon/datalake/cialab/scratch/cialab/tet/python/AttentionDeepMIL/heatmap/results_the_rest/*.png.tif';
slide_dir='/isilon/datalake/cialab/original/cialab/image_database/d00127/Scans/';
a=miceReadMap('slideMap_fixed.xlsx');

pos=[203,221,240,256,261,263,267,272,280,281,282,285,290,295,316,340,344,350,351,355,357,358,360,375,383,405,419,427,428,434,437,465,471,10,12,13,17,2,20,24,25,26,27,28,32,33,41,42,44,49,51,52,57,59,61,63,64,66,7,70,72,74,75,77,82,9,93,94,95,96,97,109,112,117,118,121,123,124,128,132,137,138,144,145,146,148,149,153,154,157,158,161,162,163,164,165,166,167,171,172,173,175,201,216,239];
neg=[102,103,104,105,106,107,108,190,191,192,193,194,195,202,218,219,220,241,254,255,257,258,259,262,264,265,266,268,269,271,273,275,276,277,278,283,284,287,288,289,292,294,296,300,301,302,303,304,305,306,308,309,310,311,312,314,315,317,318,319,320,321,324,326,327,328,329,330,331,332,333,334,335,336,337,338,339,341,345,347,348,349,353,354,359,361,362,363,364,366,367,368,369,370,371,374,376,377,378,380,381,382,385,386,389,390,391,392,393,394,395,396,397,398,399,402,403,404,407,408,409,410,411,413,414,417,418,420,421,422,423,424,425,426,429,430,431,435,438,439,440,441,443,445,446,447,448,449,450,451,453,454,455,456,457,458,460,461,464,466,467,468,469,470,474,1,100,101,11,14,16,18,19,22,23,29,34,35,36,38,39,4,43,45,46,47,48,5,50,53,54,56,6,60,65,68,71,73,76,79,8,80,81,83,84,85,88,89,91,92,99,110,113,114,115,116,119,120,122,126,127,131,134,136,139,140,141,142,143,147,151,152,159,160,168,174,176,177,179,180,181,182,183,184,185,186,187,188,189,197,198,199,200,204,205,206,208,209,211,213,217,222,223,224,225,226,227,228,229,230,232,234,235,236,237,238,242,243,244,246,247,248,249,250,251,252,253];

keys=fopen('testGillian_keys.txt','a');
dones=[""];

sf=1;
lev=0;
while true
    rng(1);
    results=dir(result_dir);
    [~,idx]=sort([results.datenum]);
    results=results(idx);
    for i=1:length(results)
        if results(i).bytes>8
            n=results(i).name;
            n=strsplit(n,'.');
            n=str2num(n{1});
            for j=1:size(a,1)
                if find(a.N{j}==n)
                    break;
                end
            end

            if find(pos==n) k='pos'; end
            if find(neg==n) k='neg'; end

            jj=j;
            mask=imread(strcat('./mouseMasks/',num2str(n),'.png'));
            p=regionprops(mask,'BoundingBox');
            for j=1:length(p)
                rfn=num2str(round(rand*10^16),16);
                if ~exist(strcat('testGillian/',rfn,'.png'))
                    r=p(j).BoundingBox;
                    imf=imfinfo(fullfile(results(i).folder,results(i).name));
                    w=imf.Width;
                    h=imf.Height;
                    scale=((w/size(mask,2))+(h/size(mask,1)))/2;
                    r=int64(r.*scale);

                    if 2^30>int64(r(3)/sf)*int64(r(4)/sf)
                        fprintf('using %i,%i\n',n,j);
                        read=0;
                        tries=0;
                        while true
                            try
                                heatmap=imread(fullfile(results(i).folder,results(i).name),'PixelRegion',{[r(2),r(2)+r(4)],[r(1),r(1)+r(3)]});
                                read=1;
                            catch
                                read=0;
                                tries=tries+1;
                            end
                            if read
                                break;
                            end
                            if tries>100
                                break;
                            end
                        end
                        if ~read
                            continue;
                        end
                        heatmap=imresize(heatmap,1/sf);
                        
                        imwrite(heatmap,strcat('testGillian/',rfn,'.png'));
                        fprintf(keys,'%s,%s\n',rfn,k);
                        dones=cat(1,dones,strcat(num2str(n),"_",num2str(j)));
                    end
                end
            end
        end
    end
end
fclose(keys);
    