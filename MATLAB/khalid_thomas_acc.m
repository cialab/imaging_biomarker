% Wow I can't believe I was able to recover this
% There are two images whose rfn I couldn't recover b/c of the weird loop
% in trying to read the image

t=fileread('testGillian_mouse_nums.csv');
t=strsplit(t,'\n');
a=[""];
b=zeros(length(t),1);
for i=1:length(t)
    c=strsplit(t{i},',');
    a(i)=string(c{1});
    b(i)=str2num(c{2});
end

d=dir('testGillian/*.png');
load('ss_labels_new.mat');
load('thomas_predict.mat','prs');
thomas_prs=prs;
load('progress.mat');
khalid_prs=prs;
clear prs;

thomas_prs_new=[];
khalid_prs_new=[];
gts=[];
for i=1:length(d)
    n=d(i).name;
    n=strsplit(n,'.');
    n=n{1};
    
    % Find image in new list
    ii=find(strcmp(a,n));
    if ~isempty(ii)
        l=label(find(b(ii)==mouseNum));
        if l~=-1
            gts=[gts l];
            thomas_prs_new=[thomas_prs_new thomas_prs(i)];
            khalid_prs_new=[khalid_prs_new khalid_prs(i)];
        end
    else
        fprintf('%s not found\n',d(i).name);
    end
end







gts=gts;
prs=thomas_prs_new;
c=confusionmat(gts,prs);
fprintf('%2.2f %2.2f %2.2f %2.2f %2.2f\n',(c(1,1)+c(2,2))/sum(c(:))*100,c(1,1)/sum(c(1,:))*100,c(2,2)/sum(c(2,:))*100,c(1,1)/sum(c(:,1))*100,c(2,2)/sum(c(:,2))*100);

% Bootstrap
acc=zeros(1000,1);
se=zeros(1000,1);
sp=zeros(1000,1);
ppv=zeros(1000,1);
npv=zeros(1000,1);
for i=1:1000
    r=randsample(614,614,'true');
    c=confusionmat(gts(r),prs(r));
    
    acc(i)=(c(1,1)+c(2,2))/sum(c(:));
    se(i)=c(1,1)/sum(c(1,:));
    sp(i)=c(2,2)/sum(c(2,:));
    ppv(i)=c(1,1)/sum(c(:,1));
    npv(i)=c(2,2)/sum(c(:,2));
end
fprintf('[%2.2f,%2.2f] [%2.2f,%2.2f] [%2.2f,%2.2f] [%2.2f,%2.2f] [%2.2f,%2.2f]\n',prctile(acc,5)*100,prctile(acc,95)*100,prctile(se,5)*100,prctile(se,95)*100,prctile(sp,5)*100,prctile(sp,95)*100,prctile(ppv,5)*100,prctile(ppv,95)*100,prctile(npv,5)*100,prctile(npv,95)*100);
