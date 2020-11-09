function makeBag(fp,y,x,mask,mouseNum,label,num_instances,ps,fg,wd)
% Makes a bag out of a slide
    
    bag=zeros(1,10,3,ps,ps,'uint8');
    if ~exist(fullfile(wd,num2str(label)))
        mkdir(fullfile(wd,num2str(label)));
    end
    bag=permute(bag,[5 4 3 2 1]);
    h5create(fullfile(wd,num2str(label),strcat(num2str(mouseNum),'.h5')),'/bag',size(bag),'Datatype','uint8');
    h5write(fullfile(wd,num2str(label),strcat(num2str(mouseNum),'.h5')),'/bag',bag);
end

