clear all;
datanames={'20161007','20161011'};
for dn = 1:length(datanames)
    dataname = datanames{dn};
    if dataname == '20161007'
        load('/home/lyg/dataset/opendata/DAD-data-10-21-2017/data/Chewie/Chewie_CO_FF_2016_10_07.mat');
    elseif dataname == '20161011'
        load('/home/lyg/dataset/opendata/DAD-data-10-21-2017/data/Chewie/Chewie_CO_FF_2016_10_11.mat');
    end

    j=1;
    x=[];
    y=[];
    trialno=[];
    pos=[];
    for i=1:size(trial_data,2)
        disp(j);
        data = trial_data(i);
        if isnan(data.idx_go_cue)
            continue;
        end
        Data(j).PMD = data.PMd_spikes(data.idx_go_cue:end,:);
        Data(j).vel = data.vel(data.idx_go_cue:end,:);
        Data(j).pos = data.pos(data.idx_go_cue:end,:);
        Data(j).trialno = ones(size(Data(j).PMD,1),1)*j;
        x=[x;Data(j).PMD];
        y=[y;Data(j).vel];
        trialno=[trialno;ones(size(Data(j).PMD,1),1)*j];
        pos = [pos;Data(j).pos];
        j=j+1;
    end
    save(['C_',dataname,'.mat'],'x','y','trialno','pos');

    load(['C_',dataname,'.mat']);
    binNum=10;
    uniq_trialno = unique(trialno);
    binned_pos = [];
    binned_pos_trialno = [];
    binned_vel = [];
    binned_pmd = [];
    binned_vel_trialno=[];
    for i=1:length(uniq_trialno)
        idx = find(trialno==uniq_trialno(i));
        pmd_data_trial= x(idx,:);
        pos_trial = pos(idx,:);
        triallen = length(idx);
        binnedTriallen  = ceil(triallen/binNum);
        binned_pos_trial=[pos_trial(1,:)];
        binned_pmd_trial=[];
        for j = 1:binnedTriallen
            end_pos_idx = j*binNum;
            if end_pos_idx>triallen
                end_pos_idx=triallen;
            end
            binned_pos_trial = [binned_pos_trial;pos_trial(end_pos_idx,:)];
            binned_pmd_trial = [binned_pmd_trial;sum(pmd_data_trial((j-1)*binNum+1:end_pos_idx,:),1)];
        end
        binned_pos = [binned_pos;binned_pos_trial];
        binned_pos_trialno = [binned_pos_trialno;ones(size(binned_pos_trial,1),1)*i];
        binned_vel = [binned_vel;binned_pos_trial(2:end,:)-binned_pos_trial(1:(end-1),:)];
        binned_pmd = [binned_pmd;binned_pmd_trial];
        binned_vel_trialno = [binned_vel_trialno;ones(size(binned_pos_trial,1)-1,1)*i];
    end
    x = binned_pmd;
    y = binned_vel/0.1;
    pos = binned_pos;
    pos_trialno = binned_pos_trialno;
    trialno = binned_vel_trialno;

    uniq_trialno = unique(pos_trialno);
    trial_ends=[];
    for i=1:length(uniq_trialno)
        idx = find(pos_trialno==uniq_trialno(i));
        trial_ends = [trial_ends;pos(idx(end),:)];
    end
    
    %% remove invalid trials
    plot(trial_ends(:,1),trial_ends(:,2),'*');
    rng(1);
    [drt_trial,drt_pos]=kmeans(trial_ends(1:end,:),8);
    for i=1:size(trial_ends,1)
        distant = sqrt(sum((trial_ends(i,:)-drt_pos).^2,2));
        min_dists(i) = min(distant);
    end
    del_idxs = find(min_dists>=1.5);
    disp(del_idxs)
    trial_ends(del_idxs,:)=[];
    for i=1:length(del_idxs)
        trial_del_idx = del_idxs(i);
        idx = find(trialno == uniq_trialno(trial_del_idx));
        y(idx,:)=[];
        x(idx,:)=[];
        trialno(idx,:)=[];
        idx = find(pos_trialno == uniq_trialno(trial_del_idx));
        pos(idx,:)=[];
        pos_trialno(idx,:)=[];
    end

    [drt_trial,drt_pos]=kmeans(trial_ends(1:end,:),8);
    plot(trial_ends(:,1),trial_ends(:,2),'*');

    drt_pos_sub_mean = drt_pos-mean(drt_pos,1);
    pos_theta = atan2(drt_pos_sub_mean(:,2),drt_pos_sub_mean(:,1))/pi*180;
    pos_theta(find(pos_theta<0))=pos_theta(find(pos_theta<0))+360;
    pos_theta_ideal=0:45:315;
    for i=1:8
        theta_res = pos_theta(i)-pos_theta_ideal;
        theta_res(theta_res>180)=360-theta_res(theta_res>180);
        theta_res(theta_res<-180)=360+theta_res(theta_res<-180);
        [~,idx] = min(abs(theta_res));
        pos_correspond_true_poss(i)=idx;
    end

    sub_no = pos_correspond_true_poss;
    drt_trial_sub = drt_trial;
    for i=1:8
        idx = find(drt_trial==i);
        drt_trial_sub(idx)=sub_no(i);
    end

    %% generate vel_drtno pos_drtno
    uniq_trialno = unique(trialno);
    drtno=[];
    pos_drtno=[];
    for i=1:length(uniq_trialno)
        idx1 = find(trialno==uniq_trialno(i));
        drtno=[drtno;ones(length(idx1),1)*drt_trial_sub(i)];
        idx2 = find(pos_trialno==uniq_trialno(i));
        pos_drtno = [pos_drtno;ones(length(idx2),1)*drt_trial_sub(i)];
    end
    
    %% remove channels with firing rate less than 0.5Hz
    mean_x=mean(x,1);
    x(:,find(mean_x<0.05))=[];
    save(['C_',dataname,'_bin100ms.mat'],'x','y','pos','pos_trialno','trialno','drtno','pos_drtno');


    %% split 5fold 
    load(['C_',dataname,'_bin100ms.mat']);
    for i=1:8
        trials = unique(trialno(find(drtno==i)));
        trialnums(i)=length(trials);
        drt_include_trial{i}=trials;
    end
    trialnums_per_fold=round(trialnums/5);
    for i=1:5
        trials_per_fold=[];
        for j=1:8
            start_idx = (i-1)*trialnums_per_fold(j)+1;
            end_idx = i*trialnums_per_fold(j);
            if i==5
                end_idx = trialnums(j);
            end
            trials_per_fold = [trials_per_fold;drt_include_trial{j}(start_idx:end_idx)];
        end
        trials_per_fold_cell{i}=trials_per_fold;
        pmd_per_fold=[];
        kin_per_fold=[];
        drtno_per_fold=[];
        trialno_per_fold=[];
        for k = 1:length(trials_per_fold)
            idx = find(trialno==trials_per_fold(k));
            pmd_per_fold=[pmd_per_fold;x(idx,:)];
            kin_per_fold=[kin_per_fold;y(idx,:)];
            drtno_per_fold=[drtno_per_fold;drtno(idx,:)];
            trialno_per_fold=[trialno_per_fold;trialno(idx,:)];
        end
        pmd_per_fold_cell{i}=pmd_per_fold;
        kin_per_fold_cell{i}=kin_per_fold;
        drtno_per_fold_cell{i}=drtno_per_fold;
        trialno_per_fold_cell{i}=trialno_per_fold;
    end
    trials=[];
    for i=1:5
        trials=[trials;trials_per_fold_cell{i}];
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

    save_dir='C_5fold_PMD/';
    if ~ exist(save_dir,'dir')
        mkdir(save_dir);
    end
    for i=1:5
        NeuralData=[];
        KinData=[];
        TrialNo=[];
        DrtNo=[];
        for j=1:3
            NeuralData=[NeuralData;pmd_per_fold_cell{trains(i,j)}];
            KinData=[KinData;kin_per_fold_cell{trains(i,j)}];
            TrialNo = [TrialNo;trialno_per_fold_cell{trains(i,j)}];
            DrtNo = [DrtNo;drtno_per_fold_cell{trains(i,j)}];
        end
        NeuralData = NeuralData';
        KinData = KinData';
        save([save_dir,dataname,'_',num2str(i),'_train.mat'],'NeuralData','KinData','TrialNo','DrtNo');

        NeuralData=pmd_per_fold_cell{vals(i)};
        KinData=kin_per_fold_cell{vals(i)};
        TrialNo = trialno_per_fold_cell{vals(i)};
        DrtNo = drtno_per_fold_cell{vals(i)};
        NeuralData = NeuralData';
        KinData = KinData';
        save([save_dir,dataname,'_',num2str(i),'_val.mat'],'NeuralData','KinData','TrialNo','DrtNo');
        NeuralData=pmd_per_fold_cell{tests(i)};
        KinData=kin_per_fold_cell{tests(i)};
        TrialNo = trialno_per_fold_cell{tests(i)};
        DrtNo = drtno_per_fold_cell{tests(i)};
        NeuralData = NeuralData';
        KinData = KinData';
        save([save_dir,dataname,'_',num2str(i),'_test.mat'],'NeuralData','KinData','TrialNo','DrtNo');
    end
end








