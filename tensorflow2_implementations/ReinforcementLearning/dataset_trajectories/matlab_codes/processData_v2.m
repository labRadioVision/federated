function processData_v2(filename)
image_target_size = [80, 88];
load(filename)
robot_movements = 5;
robot_positions = 35;
data_hexapods = cell(robot_positions,robot_movements);
pos_index = zeros(35,1);
done_pos = 30;

%data = cell(duration2,1);
counter_g = 0;
for k=1:size(fused,1)
     if isempty(fused{k,1})
         break
     else
            counter_g = counter_g + 1;
            im_grey = fused{k,1}.camera(:,:,1) + fused{k,1}.camera(:,:,2) + fused{k,1}.camera(:,:,3);
            a1 = imresize(im_grey,[NaN,image_target_size(2)]); %64 or 81
            for kq=1:image_target_size(2) 
                camera_resize(:,kq)=resample(a1(:,kq),image_target_size(1),size(a1,1)); %81 61 or 64 48
            end
            
            %fused{k,1}.camera = imrotate(fliplr(camera_resize./255.'),-90);
            fused{k,1}.camera = imrotate(camera_resize./255,180); 
            
            %%%verify image
            %imagesc(fused{k,1}.camera)
            %prompt = {'Enter position:'};
            %dlgtitle = 'Input';
            %dims = [1 35];
            %definput = {'0'};    
            %answer = inputdlg(prompt,dlgtitle,dims,definput);
            %fused{k,1}.position = str2double(answer);
            %%%%%%%%%%%%%%%%%%%%
            fused{k,1}.position = floor((k-1)/robot_movements);
            pos_index(fused{k,1}.position+1,1) = pos_index(fused{k,1}.position+1,1) + 1;
            if fused{k,1}.position == done_pos
                fused{k,1}.done = true;
            end
            data_hexapods{fused{k,1}.position+1,pos_index(fused{k,1}.position+1,1)}=fused{k,1};
            
     end
end
%fused{counter_g,1}.done = true;
%data_hexapods = fused(1:counter_g,1);
save('data_robots2.mat','data_hexapods','-v7.3')