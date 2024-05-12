%% get MUA
load('indy_20170124_01.mat');
spk=zeros(length(t),size(spikes,1));
for i=1:96
    for j = 1:5
        spk_ch = spikes{i,j};
        if ~isempty(spk_ch)
            for k=1:length(spk_ch)
                idx=find(t>spk_ch(k));
                if isempty(idx)
                    continue;
                end
                idx=idx(1);
                spk(idx,i)=spk(idx,i)+1;
            end
        end
    end
end
spk(1,:)=zeros(1,96);
save('Open2017012401_MUA.mat','spk','finger_pos');


%% bin100ms
load('Open2017012401_MUA.mat');
kin=finger_pos(:,2:3);
ratio=25;
data_len = floor(size(spk,1)/ratio);
spk_100 = zeros(data_len,96);
kin_100 = zeros(data_len,2);
for i=1:data_len
    spk_100(i,:)=sum(spk((i-1)*ratio+1:i*ratio,:),1);
    kin_100(i,:)=(kin(i*ratio,:)-kin((i-1)*ratio+1,:))/0.1;
end
NeuralData=spk_100;
KinData=kin_100;
save('indy_2017012401_100.mat','NeuralData','KinData');

%% split 5 fold;
load('indy_2017012401_100.mat');
mean_fr = mean(NeuralData,1);
idx = find(mean_fr>=0.05);
NeuralData=NeuralData(:,idx);
interval = floor(linspace(1,size(KinData,1),6));
interval(end)=interval(end)+1;
spk_folds={};
kin_folds={};
for i=1:5
    spk_folds{i}=NeuralData(interval(i):(interval(i+1)-1),:);
    kin_folds{i}=KinData(interval(i):(interval(i+1)-1),:);
end

tests=[1,2,3,4,5];
hashs=zeros(1,5);
rng(1);
while 1
    vals=randperm(5,5);
    if isempty(find((vals-tests)==0))
        break;
    end
end

trains=[];
for i=1:5
    tmp_val_test=[tests(i),vals(i)];
    train=setdiff(tests,tmp_val_test);
    disp(train)
    if isempty(trains)
        trains=train;
    else
        trains=[trains;train];
    end
end

save_dir='Open2017012401_5fold/';
if ~ exist(save_dir,'dir')
    mkdir(save_dir);
end
for i=1:5
    NeuralData=[];
    KinData=[];
    for j=1:3
        NeuralData=[NeuralData;spk_folds{trains(i,j)}];
        KinData=[KinData;kin_folds{trains(i,j)}];
    end
    NeuralData=NeuralData';
    KinData=KinData';
    save([save_dir,'2017012401_',num2str(i),'_train.mat'],'NeuralData','KinData');
    NeuralData=spk_folds{vals(i)};
    KinData=kin_folds{vals(i)};
    NeuralData=NeuralData';
    KinData=KinData';
    save([save_dir,'2017012401_',num2str(i),'_val.mat'],'NeuralData','KinData');
    NeuralData=spk_folds{tests(i)};
    KinData=kin_folds{tests(i)};
    NeuralData=NeuralData';
    KinData=KinData';
    save([save_dir,'2017012401_',num2str(i),'_test.mat'],'NeuralData','KinData');
end

