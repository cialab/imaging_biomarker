% For testing images
hms=dir('testGillian/*.png');

%prs=zeros(length(hms),1);
f=figure;
for i=1:length(hms)
    hm=imread(fullfile(hms(i).folder,hms(i).name));
    imshow(hm);
    title(num2str(i));
    
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