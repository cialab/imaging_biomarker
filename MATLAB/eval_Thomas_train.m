% For training images
d=dir('trainGillian/*/*.jp2');
[val,idx]=sort({d.name});
d=d(idx);
hes=d(1:31);
hms=d(32:end);
rng(1);
r=randperm(length(hes));
hes=hes(r);
hms=hms(r);

prs=zeros(length(hms),1);
gts=zeros(length(hms),1);
f=figure;
for i=1:length(hms)
    hm=imread(fullfile(hms(i).folder,hms(i).name));
    imshow(hm);
    title(num2str(i));
    
    s=strsplit(hms(i).folder,'/');
    s=s{end};
    if strcmp(s,'nss')
        gts(i)=0;
    end
    if strcmp(s,'ss')
        gts(i)=1;
    end
    
    while 1
        w=waitforbuttonpress;
        if strcmp(f.CurrentCharacter,'s')
            prs(i)=1;
            fprintf('received %i\n',i);
            break;
        end
        if strcmp(f.CurrentCharacter,'a')
            prs(i)=0;
            fprintf('received %i\n',i);
            break;
        end
    end
end