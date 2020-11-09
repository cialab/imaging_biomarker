function makeBag(fp,y,x,mask,mouseNum,label,num_instances,ps,fg,wd)
% Makes a bag out of a slide
    
    % Save time by computing boundaries for querty points
    boundaries=bwboundaries(mask);
    minx=size(mask,2);miny=size(mask,1);maxx=0;maxy=0;
    for i=1:length(boundaries)
        b=boundaries{i};
        if min(b(:,2))<minx
            minx=min(b(:,2));
        end
        if min(b(:,1))<miny
            miny=min(b(:,1));
        end
        if max(b(:,2))>maxx
            maxx=max(b(:,2));
        end
        if max(b(:,1))>maxy
            maxy=max(b(:,1));
        end
    end
    psx=ps*size(mask,1)/double(x);
    psy=ps*size(mask,2)/double(y);
    
    % Here's where we put things
    bag=zeros(1,num_instances,3,ps,ps,'uint8');
    n=1;
    i=1;
    while n<=num_instances
        % Query points
        rx=minx+rand*(maxx-minx);
        ry=miny+rand*(maxy-miny);
        
        % Checks if inside the mask image
        if (size(mask,2)>rx+psx) && (size(mask,1)>ry+psy)
            % Checks if inside polygons
            in=0;
            for i=1:length(boundaries)
                b=boundaries{i};
                if inpolygon(rx,ry,b(:,2),b(:,1)) && ...
                        inpolygon(rx+psx,ry,b(:,2),b(:,1)) && ...
                        inpolygon(rx,ry+psy,b(:,2),b(:,1)) && ...
                        inpolygon(rx+psx,ry+psy,b(:,2),b(:,1))
                    in=1;
                    break
                end
            end
            if in
                im=openslide_read_region(fp,round(rx*double(y)/size(mask,2)),round(ry*double(x)/size(mask,1)),ps,ps);
                p=im(:,:,2:4);

                % Check foreground percentage
                if sum(sum(rgb2gray(p)<230))/(ps*ps)>fg
                    bag(:,n,:,:,:)=permute(p,[3 2 1]);
                    n=n+1;
                end
            end
        end
        i=i+1;
    end
    if ~exist(fullfile(wd,num2str(label)))
        mkdir(fullfile(wd,num2str(label)));
    end
    bag=permute(bag,[5 4 3 2 1]);
    h5create(fullfile(wd,num2str(label),strcat(num2str(mouseNum),'.h5')),'/bag',size(bag),'Datatype','uint8');
    h5write(fullfile(wd,num2str(label),strcat(num2str(mouseNum),'.h5')),'/bag',bag);
end

