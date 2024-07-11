data_dir='dataset/ObstacleAvoidanceData/';
filenames=dir(data_dir);
j=1;
mat_files={};
for i=3:length(filenames)
    if contains(filenames(i).name,'.mat')
        mat_files{j}=filenames(i).name;
        j=j+1;
    end
end

save_dir='./';
if ~exist(save_dir)
    mkdir(save_dir)
end

for dd=1:length(mat_files)
    filename=mat_files{dd};
    load([data_dir,filename])
    trialno=TrialNo;
    spk=NeuralData;
    kin=KinData;

    tmp_filename=split(filename,'.');
    tmp_filename=tmp_filename{1};
    
    uniq_trialno=unique(trialno);
    max_trialno=length(uniq_trialno);
    
    [drtno,unique_drts,num_each_drt]=get_drtinfo(TrialNo,PosData);
    
    
    for i=1:length(unique_drts)
        trials = unique(trialno(find(drtno==unique_drts(i))));
        trialnums(i)=length(trials);
        drt_include_trial{i}=trials;
    end
    trialnums_per_fold=round(trialnums/5);
    
    for i=1:5
        trials_per_fold=[];
        for j=1:length(unique_drts)
            start_idx = (i-1)*trialnums_per_fold(j)+1;
            end_idx = i*trialnums_per_fold(j);
            if i==5
                end_idx = trialnums(j);
            end
            trials_per_fold = [trials_per_fold,drt_include_trial{j}(start_idx:end_idx)];
        end
        trials_per_fold_cell{i}=trials_per_fold;
        pmd_per_fold=[];
        kin_per_fold=[];
        drtno_per_fold=[];
        trialno_per_fold=[];
        for k = 1:length(trials_per_fold)
            idx = find(trialno==trials_per_fold(k));
            pmd_per_fold=[pmd_per_fold,spk(:,idx)];
            kin_per_fold=[kin_per_fold,kin(:,idx)];
            drtno_per_fold=[drtno_per_fold,drtno(:,idx)];
            trialno_per_fold=[trialno_per_fold,trialno(:,idx)];
        end
        pmd_per_fold_cell{i}=pmd_per_fold;
        kin_per_fold_cell{i}=kin_per_fold;
        drtno_per_fold_cell{i}=drtno_per_fold;
        trialno_per_fold_cell{i}=trialno_per_fold;
    end
    trials=[];
    for i=1:5
        trials=[trials,trials_per_fold_cell{i}];
    end
    unique(trials)
    unique(trialno)

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

    for i=1:5
        NeuralData=[];
        KinData=[];
        TrialNo=[];
        DrtNo=[];
        for j=1:3
           NeuralData=[NeuralData,pmd_per_fold_cell{trains(i,j)}];
           KinData=[KinData,kin_per_fold_cell{trains(i,j)}];
           TrialNo = [TrialNo,trialno_per_fold_cell{trains(i,j)}];
           DrtNo = [DrtNo,drtno_per_fold_cell{trains(i,j)}];
        end
        save([save_dir,tmp_filename,'_',num2str(i),'_train.mat'],'NeuralData','KinData','TrialNo','DrtNo');

        NeuralData=pmd_per_fold_cell{vals(i)};
        KinData=kin_per_fold_cell{vals(i)};
        TrialNo = trialno_per_fold_cell{vals(i)};
        DrtNo = drtno_per_fold_cell{vals(i)};
        save([save_dir,tmp_filename,'_',num2str(i),'_val.mat'],'NeuralData','KinData','TrialNo','DrtNo');

        NeuralData=pmd_per_fold_cell{tests(i)};
        KinData=kin_per_fold_cell{tests(i)};
        TrialNo = trialno_per_fold_cell{tests(i)};
        DrtNo = drtno_per_fold_cell{tests(i)};
        save([save_dir,tmp_filename,'_',num2str(i),'_test.mat'],'NeuralData','KinData','TrialNo','DrtNo');
    end
    
end
function [direction_bin,unique_drts,num_each_drt]=get_drtinfo(TrialNo,PosData)
    centroids=[[-22,40];[-14,43];[-7,40];[-14,36]];
    uniq_trialno=unique(TrialNo);
    direction_bin=zeros(1,length(TrialNo));
    directions=[];
    for i=1:length(uniq_trialno)
        idx = find(TrialNo==uniq_trialno(i));
        sd=judge_directions(PosData(:,idx(1)),centroids);
        ed=judge_directions(PosData(:,idx(end)),centroids);
        direction_bin(idx)=(sd*10+ed);
        directions=[directions,sd*10+ed];
    end
    unique_drts=unique(direction_bin);
    num_each_drt=zeros(1,length(unique_drts));
    for i=1:length(unique_drts)
        num_each_drt(i)=length(find(directions==unique_drts(i)));
    end
end
