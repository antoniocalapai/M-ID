%% anc_mIDstage2_CurateDataFrame
% Combine data from multiple sources:
% - excel files with manual labels
% - csv file with trial information
% - excel file with metadata information
disp('This scripts requires a long time to run ..')
addpath(genpath('/Users/antoninocalapai/ownCloud/Shared/Cognitive_testing'))

user = char(java.lang.System.getProperty('user.name'));
warning('OFF', 'MATLAB:table:RowsAddedExistingVars')

% add tpaths with MWorks analysis scripts from the owncloud
addpath(genpath((['/Users/' user '/ownCloud/Shared/MWorks_MatLab'])))

% add and set relevant paths from weco server
addpath(genpath('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MatlabTools'))
analysisPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/';
dataPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/data/';
dataFramePath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/dataframes/';
ownCloudPath = ['/Users/' user '/ownCloud/Shared/WeCo_mID/analysis/dataframes/'];
plotPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/plots/';
labelsPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/Labels/';
trainPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures';

%% Gather all the excel with manual labels into one table
AllLabels = [];
AllLabelFiles = dir(fullfile(labelsPath, '*.xlsx'));  % Use absolute path names
for i = 1:length(AllLabelFiles)
    disp(AllLabelFiles(i).name)
    F = readtable([labelsPath AllLabelFiles(i).name]);
    F = removevars(F,{'notes'});
    try
        F = removevars(F,{'app_version'});
    catch
    end
    AllLabels = [AllLabels;F];
end

AllLabels.filename = string(AllLabels.filename);
AllLabels.manual_label = string(lower(AllLabels.manual_label));
AllLabels.user = string(AllLabels.user);
AllLabels.ML_id = string(nan(size(AllLabels,1),1));

% fix a typo on alw's label
AllLabels.manual_label(strcmp(AllLabels.manual_label,'alv')) = {'alw'};

% remove trial information from picture name
AllLabels.picID = repmat({''}, size(AllLabels,1), 1);
for i = 1:size(AllLabels,1)
    disp(AllLabels.filename{i}(1:end-10))
    AllLabels{i, 'picID'} = {AllLabels.filename{i}(1:end-10)};
end
AllLabels.picID = string(AllLabels.picID);

%% Make a list of all pictures used for training
AllTrainDirs = {'/AlwCla/train/', ...
                '/DerElm/train/', ...
                '/EdgSun/train/', ...
                '/IggJos/train/', ...
                '/RodCojZac/train/', ...
                '/HeiLotPanSan/train/'};
            
AllTrainPics = [];
for i = 1:length(AllTrainDirs)
    disp(AllTrainDirs(i))
    tempPath = [trainPath, AllTrainDirs{i},'**/*.jpg'];
    tempPics = dir(tempPath);
    
    tempPics = struct2table(tempPics);
    tempPics.name = categorical(tempPics.name);
    AllTrainPics = vertcat(AllTrainPics, tempPics);
end

AllTrainPics.picID = repmat({''}, size(AllTrainPics,1), 1);
for i = 1:size(AllTrainPics,1)
    disp(i)
    s = char(AllTrainPics.name(i));
    AllTrainPics{i, 'picID'} = cellstr(s(1:end-10));
end

%% Gather metadata from the (most recent) excel file (manually compiled from animal's protocols)
d = dir([dataFramePath '*metadata*']);
dd = zeros(length(d),1);
for j = 1:length(d)
    dd(j,1) = datenum(d(j).date);
end
[~, mostRecent] = max(dd);

metadata = readtable([dataFramePath d(mostRecent).name]);

%% Gather the (most recent) csv with trial information from mworks data files
% Trials information for stage 1
d = dir([dataFramePath '*stage1_TrialsDataFrame*']);
dd = zeros(length(d),1);
for j = 1:length(d)
    dd(j,1) = datenum(d(j).date);
end
[~, mostRecent] = max(dd);

DATA = readtable([dataFramePath d(mostRecent).name]);
DATA.experiment = categorical(DATA.experiment);
DATA.date = categorical(DATA.date);
DATA.group = categorical(DATA.group);
DATA.roundTrip = zeros(size(DATA,1),1);

% Trials information for stage 2
d = dir([dataFramePath '*stage2_TrialsDataFrame*']);
dd = zeros(length(d),1);
for j = 1:length(d)
    dd(j,1) = datenum(d(j).date);
end
[~, mostRecent] = max(dd);

DATA2 = readtable([dataFramePath d(mostRecent).name]);
DATA2.experiment = categorical(DATA2.experiment);
DATA2.date = categorical(DATA2.date); 
DATA2.group = categorical(DATA2.group);

try
    DATA2 = removevars(DATA2,{'manual_label'});
end

try
    DATA = removevars(DATA,{'manual_label'});
end

% Concatenate the trials dataframes
DATA = vertcat(DATA,DATA2);

% Add columns
DATA.manual_label = zeros(size(DATA,1),1);
DATA.manual_label = categorical(DATA.manual_label);

% Initialize empty columns for metadata
DATA.age = zeros(size(DATA,1),1);
DATA.dominance = zeros(size(DATA,1),1);
DATA.implant = zeros(size(DATA,1),1);
DATA.weigth = zeros(size(DATA,1),1);
DATA.picID = repmat({''}, size(DATA,1), 1);
DATA.used_in_training = zeros(size(DATA,1),1);

%% Combine manual labels and experiment data
for i = 1:size(DATA,1)
    disp(size(DATA,1) - i)
    pic_name = DATA.filename{i}(1:end-10);
    if ismember(pic_name,AllLabels.picID)        
        l = string(AllLabels.manual_label(AllLabels.picID == pic_name));
        DATA{i, 'manual_label'} = l(1);
    else
        DATA{i, 'manual_label'} = "NotFound";
    end
    
    if sum(metadata.animalID == DATA{i, 'manual_label'})
        % compute age and assign it to table
        birthday = metadata{metadata.animalID == DATA{i, 'manual_label'}, 'birthdate'};
        age = round(double(datenum(datetime('today')) - datenum(datetime(birthday))) / 365);
        DATA{i, 'age'} = age;
        
        % assign dominance
        DATA{i, 'dominance'} = metadata{metadata.animalID == DATA{i, 'manual_label'}, 'dominance'};
        
        % assign implant
        DATA{i, 'implant'} = string(metadata{metadata.animalID == DATA{i, 'manual_label'}, 'implant'}) == "yes";
        
        % assign weight
        DATA{i, 'weigth'} = metadata{metadata.animalID == DATA{i, 'manual_label'}, 'lastWeight'};
    end
    DATA{i, 'picID'} = {DATA.filename{i}(1:end-10)};
    
    % check if the picture was used during training
    if ismember(pic_name,AllTrainPics.picID)
        DATA{i, 'used_in_training'} = 1;
    end
    
end

%% Save the resulting curated dataframe with the curation date
writetable(DATA, string(dataFramePath) + 'mID_stage1&2_CuratedDataFrame_' + string(yyyymmdd(datetime('today'))) + '.csv')
writetable(DATA, string(ownCloudPath) + 'mID_stage1&2_CuratedDataFrame_' + string(yyyymmdd(datetime('today'))) + '.csv')

%% Test 
% P = DATA(DATA.manual_label ~= "NotFound" & ...
%     DATA.manual_label ~= "null",:);
% 
% P.ML_label = categorical(P.ML_label);
% P.correct = P.ML_label == P.manual_label;
% P.group = categorical(P.group);
% P.date = string(P.date);
% 
% P.time_of_day = datestr(datenum(string(P.time_of_day),'HHMMSS'),13);
% P.time_of_day = datetime(P.time_of_day,'Format','HH:mm:ss');
% P.hour = dateshift(P.time_of_day, 'start', 'hour');
% P.session = zeros(size(P,1),1);
% 
% % Remove one redundant session from Edgar and Sunny
% P(P.group == 'edgSun' & P.date == '20211228',:) = [];
% 
% all_groups = unique(P.group);
% for i = 1:size(all_groups,1)
%     temp = P(P.group == all_groups(i),:);
%     all_dates = unique(temp.date);
%     for j = 1:size(all_dates,1)
%         P{P.group == all_groups(i) & P.date == all_dates(j), 'session'} = j;
%     end
% end
% 
% % Plot for figure
% P.hour = datetime(P.hour,'Format','HH');
% P.time_of_day = double(string(P.hour));
% 
% groups = unique(P.group);
% sessions = unique(P.session);
% fig = figure('Position',[100 100 200*length(sessions) 250*length(groups)]);
% 
% g = gramm('x',categorical(P.hour),'y',P.trial, 'color', P.manual_label);
% g.facet_grid(P.group,P.session, 'scale','free_y');
% g.stat_bin('geom','stacked_bar');
% 
% g.axe_property('XGrid','off','Ygrid','on');
% g.set_names('x','Time of Day ','y','Trials', 'color','Animals', ...
%     'column','session','row', '');
% g.set_text_options('base_size',10,...
%     'label_scaling',1.5,...
%     'legend_scaling',1.5,...
%     'legend_title_scaling',1.5,...
%     'facet_scaling',1.5,...
%     'title_scaling',1.5);
% 
% g.draw();
% g.redraw(0.01);
