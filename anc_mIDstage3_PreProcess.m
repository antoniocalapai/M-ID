%function anc_mIDstage1_PreProcess(ExtractAllSessions, ExtractPictures)
%% This function extracts trial information and pictures from each data file of the Monkey ID project
% Two arguments need to be provided:
% - ExtractPictures: if set to 1 all the snapshots taken during the task will be saved on the WeCo server
% - ExtractAllSessions: if set to 1 all the sessions are extracted, regardless if thet already have been processed
%
% Usage example:
% anc_mIDv2_extract_DATA_Pics(1, 1) -> will process all the data file, extracting and eventually replacing all the valid pictures
% anc_mIDv2_extract_DATA_Pics(1, 0) -> will process and extract the pictures and trial inancformation only of new data files
% anc_mIDv2_extract_DATA_Pics(0, 1) -> will process trial information only of new data files, and will extract its pictures
% anc_mIDv2_extract_DATA_Pics(0, 0) -> will process trial information only of new data files, without extracting its pictures
% ===========================================================================================================

ExtractPictures = 1;
ExtractAllSessions = 0;

%% Automatically detect the user on macOS
user = char(java.lang.System.getProperty('user.name'));
warning('OFF', 'MATLAB:table:RowsAddedExistingVars')

% add tpaths with MWorks analysis scripts from the owncloud
addpath(genpath((['/Users/' user '/ownCloud/Shared/MWorks_MatLab'])))

% add and set relevant paths from weco server
pathName = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/photos/notLabled/';
dataPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/rawData/';
dataframePath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/dataframes/';
scriptsPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/scripts/';
addpath(genpath(scriptsPath))

datafiles = dir([dataPath '*stage3*']);

% set up the dataframe
colNames = {'ML_label', 'manual_label', 'outcome', 'date', 'timestamp', 'trial', 'group', 'xbi', ...
    'time_of_day', 'roundTrip', 'experiment', 'filename', 'trial_start', 'trial_end'};

d = dir([dataframePath '*mID_stage3_Trials*']);
dd = zeros(length(d),1);
for j = 1:length(d)
    dd(j,1) = datenum(d(j).date);
end
[~, i] = max(dd);

if isempty(dd) || ExtractAllSessions
    T = cell2table(cell(0,length(colNames)), 'VariableNames', colNames);
else
    T = readtable([dataframePath d(i).name]);
    T.manual_label = nan(size(T,1),1);
    T = convertvars(T,string(colNames(:))','categorical');
end

%% cycle through each datafile
for i = 1:length(datafiles)
    % parse the datafile name to extract group information
    pattern = split(datafiles(i).name, '_');
    group_session = [string(pattern{1}) + '_' + string(pattern{2})];
    groupPath = [string(pathName) + string(pattern(1)) + '/valid/' + pattern{2} + '/'];
    
    allfiles = dir([char(groupPath),'*.jpg']);
    allfiles = string({allfiles(:).name}');
    
    groupName = pattern{1};
    folderName  = pattern{2};
    
    % check if the folder exists as indication if the data has already been extracted
    ExtractSession = ~isfolder([pathName groupName '/' folderName '/']);
    
    % if the folder does not exist or if all sessions have been set to be extracted
    if ExtractSession == 1 || ExtractAllSessions == 1
        disp(datafiles(i).name)
        
        % if pictures need to be extracted
        if ExtractPictures == 1
            mkdir([pathName groupName '/' folderName '/'])
        end
        
        % extract the first minute of the data file to extract the experiment time
        fileName = [dataPath datafiles(i).name];
        data = MW_readFile(fileName, 'include', {'#announceMessage'}, ...
            'stepSize', 60000000 * 600,...
            'debugLevel', 0,...    
            '~renumberTrials', ...
            '~typeOutcomeCheck', ...
            '~cleanTrialBorders');
        
        hurz = data.value(data.event == 'LOG_message');
        dateTimeStr = hurz{contains(hurz, 'Current date/time is')};
        
        if iscell(dateTimeStr)
            dateTimeStr = dateTimeStr{1};
        end
        
        dateTimeStr = replace(dateTimeStr, 'CET ', '');
        dateTimeStr = replace(dateTimeStr, 'CEST ', '');
        dateTimeStr = dateTimeStr(22:length(dateTimeStr));
        dateTimeValue = datetime(dateTimeStr, 'InputFormat', 'eee MMM d HH:mm:ss yyyy');
        datetime.setDefaultFormats('default', 'yyyyMMdd_HHmmss');
        dateTimeStamp = data.time(contains(data.value(data.event == 'LOG_message'), 'Current date/time is'));
        dateTimeStamp = dateTimeStamp(1);
        
        % extract all needed parameters from the experiment
        params = {'TRIAL_', 'IO_midImage', 'IO_start'};
        data = MW_readFile(fileName, 'include', params);
        
        trial.start = data.time(data.event == 'TRIAL_start');
        trial.end = data.time(data.event == 'TRIAL_end');
        
        % create an empty dataframe to append to the main dataframe
        DATA = cell2table(cell(0,length(colNames)), 'VariableNames', colNames);
        
        % cycle through each trial and fill the dataframe row by row, each row represent one trial
        for k = 1:length(trial.start)
            
            % create a new row to be appended to the dataframe
            nRow = cell2table(cell(1,length(colNames)), 'VariableNames', colNames);
            
            % extract the trial outcome
            outcome = data.value(data.time > trial.start(k) & data.time < trial.end(k) & data.event == 'TRIAL_outcome');
            
            % extract the picture information
            image = data.value(data.time > trial.start(k) & data.time < trial.end(k) & data.event == 'IO_midImage');
            imageTime = data.time(data.time > trial.start(k) & data.time < trial.end(k) & data.event == 'IO_midImage');
            
            % if the trial was triggered by the animal and not automatically
            if ~isempty(outcome) && ~isempty(image)
                if strcmp(outcome{1}, 'hit') || strcmp(outcome{1}, 'noAnswer') 
                    % Define the picture name
                    pic_name = [pathName groupName '/' folderName '/' groupName '_' sprintf('%s', string(dateTimeValue + (imageTime(1) - dateTimeStamp) / 1000000 / 86400)) '_' sprintf('%05d', k) '.jpg'];
                    timeOfDay = split(pic_name,'_');
                    timeOfDay = timeOfDay{3};
                    
                    % save the picture on the weco server
                    if ExtractPictures == 1
                        disp(['Extract picture: ' pic_name(end-41:end)]);
                        fileID = fopen(pic_name,'w');
                        fwrite(fileID, image{1});
                        fclose(fileID);
                    end
                    
                    % assign the name of the group as monkey label
                    nRow.ML_label = string(groupName);
                    % do not calculate the round trip time
                    nRow.roundTrip = "NA";
                    % set the experiment as "training"
                    nRow.experiment = "training";
                    
                    
                    % store the timestamp and the time of the day in different columns
                    nRow.timestamp = imageTime(1);
                    nRow.time_of_day = string(timeOfDay);
                    
                    % store group, trial, outcome, xbi, date, and filename of the pictures
                    nRow.group = string(groupName);
                    nRow.trial = k;
                    nRow.outcome = string(outcome{1});
                    nRow.xbi = string(pattern(4));
                    
                    datetimeSplit = split(string(dateTimeValue),'_');
                    nRow.date = datetimeSplit(1);
                    
                    filenameSplit = split(pic_name, '/');
                    nRow.filename = string(filenameSplit{end});
                    
                    % store trials start and end times
                    nRow.trial_start = categorical(trial.start(k));
                    nRow.trial_end = categorical(trial.end(k));
                    
                    % add the information to the main table
                    DATA = vertcat(DATA,nRow);
                end
            end
        end
        
        % if no manual labels exist yet for this file, initiate an empty row
        DATA.manual_label = nan(size(DATA,1),1);
        DATA = convertvars(DATA,'manual_label','categorical');
        
        % for convenience and ease of use convert all columns to categorical values
        DATA = convertvars(DATA,string(colNames(:))','categorical');
        T = vertcat(T,DATA);
        
        % save the dataframe
        writetable(T, string(dataframePath) + 'mID_stage3_TrialsDataFrame_' + string(yyyymmdd(datetime('today')))+ '.csv')
        disp('Data saved')
    end
end
%end