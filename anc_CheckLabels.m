% MW_extractImages_anc
user = char(java.lang.System.getProperty('user.name'));
warning('OFF', 'MATLAB:table:RowsAddedExistingVars')

% add tpaths with analysis scripts
pathName = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/photos/notLabled/';
dataPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/rawData/';

% set paths where the data is store
datafiles = dir([dataPath '*mIDstage2*']);

colNames = {'pic_id', 'manual_label', 'ML_label', 'date', 'time', 'trial', 'xbi'};
DATA = cell2table(cell(0,length(colNames)), 'VariableNames', colNames);

% Extract pictures into separate folders
for i = 1:length(datafiles)
    
    fileInfo = split(datafiles(i).name, '_');
    group = fileInfo{1};
    session_date = fileInfo{2};
    experiment_name = fileInfo{3};
    
    TABLE_name = (['/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/_database/' group '_' session_date '.xlsx']);
    if isfile(TABLE_name)
        T = readtable(TABLE_name);
        T.filename = string(T.filename);
        T.manual_label = string(lower(T.manual_label));
        T.user = string(T.user);
        T.notes = string(T.notes);
        T.ML_id = string(nan(size(T,1),1));
        
        
        pattern = split(datafiles(i).name, '_');
        group_session = [string(pattern{1}) + '_' + string(pattern{2})];
        groupPath = [string(pathName) + string(pattern(1)) + '/valid/' + pattern{2} + '/'];
        
        allfiles = dir([char(groupPath),'*.jpg']);
        allfiles = string({allfiles(:).name}');
        
        ExtractSession = sum(contains(pattern, 'mIDstage2'));
        
        if ExtractSession == 1
            fileName = [dataPath datafiles(i).name];
            data = MW_readFile(fileName, 'include', {'#announceMessage'}, 'stepSize', 60000000 * 600);
            
            hurz = data.value(data.event == 'LOG_message');
            dateTimeStr = hurz{contains(hurz, 'Current date/time is')};
            if iscell(dateTimeStr)
                dateTimeStr = dateTimeStr{1};
            end
            
            dateTimeStr = replace(dateTimeStr, 'CET ', '');
            dateTimeStr = dateTimeStr(22:length(dateTimeStr));
            dateTimeValue = datetime(dateTimeStr, 'InputFormat', 'eee MMM d HH:mm:ss yyyy');
            datetime.setDefaultFormats('default', 'yyyyMMdd_HHmmss');
            dateTimeStamp = data.time(contains(data.value(data.event == 'LOG_message'), 'Current date/time is'));
            dateTimeStamp = dateTimeStamp(1);
            
            params = {'TRIAL_', '#announceMessage', 'IO_midImage', 'IO_midIdentifier'};
            data = MW_readFile(fileName, 'include', params);
            
            % cycle through each trial
            trial.start = data.time(data.event == 'TRIAL_start');
            trial.end = data.time(data.event == 'TRIAL_end');
            
            groupName = pattern{1};
            folderName  = pattern{2}; % date
            
            if isfolder([pathName groupName '/' folderName '/'])
                
                imgTimes = data.time(data.event == 'IO_midImage');
                
                name = split(fileName, '.');
                
                for trialCX = 1:length(trial.start)
                    nRow = cell2table(cell(1,length(colNames)), 'VariableNames', colNames);
                    
                    outcome = data.value(data.time > trial.start(trialCX) & data.time < trial.end(trialCX) & data.event == 'TRIAL_outcome');
                    image = data.value(data.time > trial.start(trialCX) & data.time < trial.end(trialCX) & data.event == 'IO_midImage');
                    imageTime = data.time(data.time > trial.start(trialCX) & data.time < trial.end(trialCX) & data.event == 'IO_midImage');
                    imageID = data.value(data.time > trial.start(trialCX) & data.time < trial.end(trialCX) & data.event == 'IO_midIdentifier');
                    
                    if strfind(outcome{1}, 'hit') % || strfind(outcome{1}, 'noAnswer')
                        pic_name = [groupName '_' sprintf('%s', string(dateTimeValue + (imageTime(1) - dateTimeStamp) / 1000000 / 86400)) '_' sprintf('%05d', trialCX) '.jpg'];
                        
                        try
                            nRow.pic_id = string(pic_name);
                            nRow.manual_label = T.manual_label(T.filename == pic_name);
                            nRow.ML_label = string(lower(imageID));
                            nRow.trial = trialCX;
                            nRow.xbi = string(pattern(4));
                            
                            datetimeSplit = split(string(dateTimeValue),'_');
                            nRow.date = datetimeSplit(1);
                            nRow.time = datetimeSplit(2);
                            
                            DATA = vertcat(DATA,nRow);
                        catch
                            disp(['picture not found in datafile ' pic_name])
                        end
                    end
                end
            end
        end
    end
end

