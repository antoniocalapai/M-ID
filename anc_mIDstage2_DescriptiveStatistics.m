%% anc_mIDstage2_DescriptiveStatistics
addpath(genpath('/Users/antoninocalapai/ownCloud/Shared/Cognitive_testing'))

% Compute the performance of the ML models
user = char(java.lang.System.getProperty('user.name'));
warning('OFF', 'MATLAB:table:RowsAddedExistingVars')

% add tpaths with MWorks analysis scripts from the owncloud
addpath(genpath((['/Users/' user '/ownCloud/Shared/MWorks_MatLab'])))

% add and set relevant paths from weco server
addpath(genpath('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MatlabTools'))
analysisPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/analysis/';
dataPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/analysis/data_analysis/dataframes/';
plotPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/analysis/data_analysis/Matlab/plots/';
labelsPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/Labels/';

% load the dataframe
d = dir([dataPath '*stage1&2_CuratedDataFrame*']);
dd = zeros(length(d),1);
for j = 1:length(d)
    dd(j,1) = datenum(d(j).date);
end
[~, i] = max(dd);

DATA = readtable([dataPath d(i).name]);

% Filter out null and errors
P = DATA(DATA.manual_label ~= "NotFound" & ...
    DATA.manual_label ~= "null",:);

P.ML_label = categorical(P.ML_label);
P.correct = P.ML_label == P.manual_label;
P.group = categorical(P.group);

P.time_of_day = datestr(datenum(string(P.time_of_day),'HHMMSS'),13);
P.time_of_day = datetime(P.time_of_day,'Format','HH:mm:ss');
P.hour = dateshift(P.time_of_day, 'start', 'hour');
P.session = zeros(size(P,1),1);

% Remove one redundant session from Edgar and Sunny
P(P.group == 'edgSun' & P.date == 20211228,:) = [];

all_groups = unique(P.group);
for i = 1:size(all_groups,1)
    temp = P(P.group == all_groups(i),:);
    all_dates = unique(temp.date);
    for j = 1:size(all_dates,1)
        P{P.group == all_groups(i) & P.date == all_dates(j), 'session'} = j;
    end
end

%% Plot for figure
P.hour = datetime(P.hour,'Format','HH');
P.time_of_day = double(string(P.hour));

groups = unique(P.group);
sessions = unique(P.session);
fig = figure('Position',[100 100 200*length(sessions) 250*length(groups)]);

g = gramm('x',categorical(P.hour),'y',P.trial, 'color', P.manual_label);
g.facet_grid(P.group,P.session, 'scale','free_y');
g.stat_bin('geom','stacked_bar');

g.axe_property('XGrid','off','Ygrid','on');
g.set_names('x','Time of Day ','y','Trials', 'color','Animals', ...
    'column','session','row', '');
g.set_text_options('base_size',10,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);

g.draw();
g.redraw(0.01);
filename_save = 'mIDstage1&2_Descriptive_A';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%% Panel B: trial per group across hours
clear g;
T = groupsummary(P,{'manual_label', 'session', 'hour', 'group', 'experiment'},'sum','correct');
T.hour = datetime(T.hour,'Format','HH');
T.time_of_day = double(string(T.hour));

animals = unique(T.manual_label);
fig = figure('Position',[100 100 750 150*length(animals)]);
g = gramm('x',categorical(T.hour),'y',T.GroupCount);
g.facet_grid(T.manual_label,T.experiment, 'scale','free_y');
g.set_order_options('column',sort(unique(categorical(T.experiment)),'descend')')

g.geom_point('dodge',1)
g.stat_glm('geom','line')
g.set_names('x','Time of Day ','y','Trials', 'color','Animals', 'column','','row', '');

g.no_legend();
g.axe_property('XGrid','on','Ygrid','on')
g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);
g.draw();
g.redraw(0.01);

set([g.results.stat_glm.line_handle],'Color',[0 0 0]);

filename_save = 'mIDstage1&2_Descriptive_B';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%%
T = groupsummary(P,{'session', 'hour', 'experiment'},'sum','correct');
T.performance = T.sum_correct ./ T.GroupCount;
T.hour = datetime(T.hour,'Format','HH');
T.time_of_day = double(string(T.hour));
clear g

fig = figure('Position',[100 100 600 300]);
g = gramm('x',categorical(T.hour),'y',T.GroupCount);
g.facet_grid([],T.experiment, 'scale','free');
g.stat_summary('geom',{'bar' 'black_errorbar'},'type','quartile', ...
    'dodge',0,'width',0.9,'setylim',true);
g.set_names('x','Time of Day','y','Trials','color','Groups');

g.no_legend();
%g.set_title('Model performance over hours');
g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1,...
    'title_scaling',1.5);
g.draw();
set([g.results.stat_summary.bar_handle],'FaceColor',[0.5 0.5 0.5]);

filename_save = 'mIDstage1&2_Descriptive_A_supp1';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%% Plot for figure
P = sortrows(P,'group');
T = groupsummary(P,{'manual_label', 'session', 'experiment'});
T = sortrows(T,'experiment','descend');

fig = figure('Position',[100 100 600 300]);
g(1,1) = gramm('x',categorical(T.manual_label),'y',T.GroupCount, 'color', T.experiment);
g(1,1).stat_summary('geom',{'bar' 'black_errorbar'},'type','quartile', ...
    'dodge',0.9,'width',0.9,'setylim',true);

g(1,1).axe_property('XGrid','off','Ygrid','on');
g(1,1).set_names('x','Animal ','y','Trials');
g(1,1).no_legend();
g(1,1).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);

g(1,2) = gramm('x', T.experiment, 'y', T.GroupCount, 'color', T.experiment);
g(1,2).geom_point()
g(1,2).stat_violin('normalization','count','width',1,'fill','transparent');
g(1,2).axe_property("YTickLabel",[])
g(1,2).axe_property('XGrid','off','Ygrid','on');
g(1,2).set_names('x','Animal ','y','', 'color', 'Exp.');

g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);


g.draw();
g.redraw(0.05);

filename_save = 'mIDstage1&2_Descriptive_A_supp2';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%%
T = groupsummary(P,{'session', 'hour', 'experiment'},'sum','correct');
T.performance = T.sum_correct ./ T.GroupCount;
T.hour = datetime(T.hour,'Format','HH');
T.time_of_day = double(string(T.hour));
clear g

fig = figure('Position',[100 100 600 300]);
g = gramm('x',categorical(T.hour),'y',T.GroupCount);
g.facet_grid([],T.experiment, 'scale','free');
g.stat_summary('geom',{'bar' 'black_errorbar'},'type','quartile', ...
    'dodge',0,'width',0.9,'setylim',true)
g.set_names('x','Time of Day','y','Trials','color','Groups','column','');
g.no_legend();
g.axe_property('XGrid','on','Ygrid','on');
g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);
g.draw();
filename_save = 'mIDstage1&2_Descriptive_C';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')


%%
T = groupsummary(P,{'manual_label', 'session', 'experiment'},'sum','correct');
T.performance = T.sum_correct ./ T.GroupCount;
clear g

fig = figure('Position',[100 100 600 300]);
g = gramm('x',categorical(T.session),'y',T.GroupCount);
g.facet_grid([],T.experiment, 'scale','free');
g.stat_summary('geom',{'bar' 'black_errorbar'},'type','quartile', ...
    'dodge',0,'width',0.9,'setylim',true)
g.set_names('x','Sessions','y','Trials','color','Groups','column','');
g.no_legend();
g.axe_property('XGrid','on','Ygrid','on');
g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);
g.draw();
filename_save = 'mIDstage1&2_Descriptive_D';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%%
% T = groupsummary(groupsummary(P,{'manual_label', 'session'}),'manual_label', 'median', 'GroupCount');

% %% Plot for figure
% P.hour = datetime(P.hour,'Format','HH');
% P.time_of_day = double(string(P.hour));
%
% groups = unique(P.group);
% fig = figure('Position',[100 100 500 200*length(groups)]);
%
% g = gramm('x',categorical(P.hour),'y',P.trial);
% g.facet_grid(P.group,P.experiment, 'scale','free','row_labels',false);
% g.stat_bin('geom','stacked_bar');
%
%
% g.axe_property('XGrid','off','Ygrid','on');
% g.set_names('x','Time of Day ','y','Trials', 'color','Animals', 'column','');
% g.set_text_options('base_size',10,...
%     'label_scaling',1.5,...
%     'legend_scaling',1.5,...
%     'legend_title_scaling',1.5,...
%     'facet_scaling',1.5,...
%     'title_scaling',1.5);
%
% g.draw();
% set([g.results.stat_bin.bar_handle],'FaceColor',[0.7 0.7 0.7])
% filename_save = 'mIDstage1&2_Descriptive_B';
% g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

