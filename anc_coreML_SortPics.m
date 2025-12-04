function anc_coreML_SortPics(groupName)
groupName = 'RodCojZac';
searchString = strcat('*', groupName, '*');

% Automatically detect the user on macOS
user = char(java.lang.System.getProperty('user.name'));
warning('OFF', 'MATLAB:table:RowsAddedExistingVars')

% add tpaths with MWorks analysis scripts from the owncloud
addpath(genpath((['/Users/' user '/ownCloud/Shared/MWorks_MatLab'])))
labelsPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/Labels/';
picsPath = '/Users/acalapai/Desktop/notLabeled/' + string(groupName) + '/';
destPath = '/Users/acalapai/Desktop/';

AllLabels = [];
AllLabelFiles = dir(fullfile(labelsPath, searchString));  % Use absolute path names
for i = 1:length(AllLabelFiles)
    F = readtable([labelsPath AllLabelFiles(i).name]);
    F = removevars(F,{'notes'});
    try
        F = removevars(F,{'app_version'});
    catch
    end
    
    temp = split(AllLabelFiles(i).name,'_');
    temp = split(temp(2),'.');
    
    allPath = strcat(picsPath, temp{1}, '/');
    F.absPath = repmat(allPath,size(F,1),1);
    
    AllLabels = [AllLabels;F];
end

AllLabels.filename = string(AllLabels.filename);
AllLabels.manual_label = string(lower(AllLabels.manual_label));
AllLabels.user = string(AllLabels.user);
AllLabels.ML_id = string(nan(size(AllLabels,1),1));

AllLabels.picID = repmat({''}, size(AllLabels,1), 1);
for i = 1:size(AllLabels,1)
    AllLabels{i, 'picID'} = {AllLabels.filename{i}(1:end-10)};
end
AllLabels.picID = string(AllLabels.picID);

%%
T = AllLabels;
T(ismember(T.manual_label,"null"),:)=[];
T.train = zeros(size(T,1),1);
T.test = zeros(size(T,1),1);

animals = unique(T.manual_label);

N = groupsummary(T,{'manual_label'});
N = min(N.GroupCount) - 144;

for i = 1:length(animals)
    x = animals(i);
    P = T(T.manual_label == x,:);
    toMove = randsample(P.filename,N);
    
    for j = 1:size(P,1)
        if ismember(P{j,'filename'}, toMove)            
            S = strcat(P{j,'absPath'},P{j,'filename'});
            D = strcat(destPath, groupName, '/train/', x, '/', P{j,'filename'});            
            copyfile(S,D)
            
        else
            S = strcat(P{j,'absPath'},P{j,'filename'});
            D = strcat(destPath, groupName, '/test/', x, '/', P{j,'filename'});            
            copyfile(S,D)
         end
    end  
    
end

end











