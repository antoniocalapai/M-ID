%% anc_mIDstage2_MLmodelPerformance
addpath(genpath('/Users/antoninocalapai/ownCloud/Shared/Cognitive_testing'))

% Compute the performance of the ML models
user = char(java.lang.System.getProperty('user.name'));
warning('OFF', 'MATLAB:table:RowsAddedExistingVars')

% add tpaths with MWorks analysis scripts from the owncloud
addpath(genpath((['/Users/' user '/ownCloud/Shared/MWorks_MatLab'])))

% add and set relevant paths from weco server
addpath(genpath('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MatlabTools'))
analysisPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/analysis/';
dataPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/analysis/data/';
plotPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/analysis/plots/';
labelsPath = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/Labels/';

% load the dataframe
d = dir([dataPath '*CuratedDataFrame*']);
dd = zeros(length(d),1);
for j = 1:length(d)
    dd(j,1) = datenum(d(j).date);
end
[~, i] = max(dd);

DATA = readtable([dataPath d(i).name]);

%% Filter dataframe, format date, hour, and give sequential number to sessions
P = DATA(DATA.manual_label ~= "NotFound" & ...
    DATA.manual_label ~= "null",:);

P.ML_label = categorical(P.ML_label);
P.correct = P.ML_label == P.manual_label;
P.group = categorical(P.group);

P.time_of_day = datestr(datenum(string(P.time_of_day),'HHMMSS'),13);
P.time_of_day = datetime(P.time_of_day,'Format','HH:mm:ss');
P.hour = dateshift(P.time_of_day, 'start', 'hour');
P.session = zeros(size(P,1),1);

P(P.group == 'edgSun' & P.date == 20211228,:) = [];
P = P(categorical(P.experiment) == 'testing',:);

all_groups = unique(P.group);
for i = 1:size(all_groups,1)
    temp = P(P.group == all_groups(i),:);
    all_dates = unique(temp.date);
    for j = 1:size(all_dates,1)
        P{P.group == all_groups(i) & P.date == all_dates(j), 'session'} = j;
    end
end

%% Plot model performance
T = groupsummary(P,{'manual_label', 'session', 'hour', 'group'},'sum','correct');
T.performance = T.sum_correct ./ T.GroupCount;
T.hour = datetime(T.hour,'Format','HH');
T.time_of_day = double(string(T.hour));

clear g
groups = categorical(unique(T.group));
fig = figure('Position',[100 100 1200 200*length(groups)]);
g = gramm('x',categorical(T.hour),'y',T.performance,'color',categorical(T.group));
g.facet_grid(T.session,T.group, 'scale','free','column_labels',false);
g.stat_summary('geom','bar', 'dodge',0,'width',0.9)
g.set_names('x',[],'y','Model Perf.','color','Groups','row', 'Session');
g.axe_property('Ylim',[.25 1],'XGrid','off','Ygrid','on');
g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);
g.geom_hline('yintercept', mean(T.performance), 'style', 'k--')
g.draw();

filename_save = 'mIDstage2_ModelPerformance_A';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')


T = groupsummary(P,{'manual_label', 'session', 'hour'},'sum','correct');
T.performance = T.sum_correct ./ T.GroupCount;
T.hour = datetime(T.hour,'Format','HH');
T.time_of_day = double(string(T.hour));

clear g
fig = figure('Position',[100 100 600 300]);
g = gramm('x',T.performance);
g.stat_bin('geom','stacked_bar','fill','transparent','nbins',50); %histogram
g.geom_vline('xintercept',mean(T.performance),'style','k--');
g.geom_vline('xintercept',median(T.performance),'style','r--');
g.set_names('x','Accuracy','y','#');
g.set_text_options('base_size',18,...
    'label_scaling',1.5,...
    'legend_scaling',1,...
    'legend_title_scaling',1,...
    'facet_scaling',1,...
    'title_scaling',1.5);

g.draw();
set([g.results.stat_bin.bar_handle],'FaceColor',[0.5 0.5 0.5]);
set([g.results.stat_bin.bar_handle],'EdgeColor',[0.5 0.5 0.5]);

disp('IQR ' + string(iqr(T.performance)))
disp('min acc ' + string(min(T.performance)))

filename_save = 'mIDstage2_ModelPerformance_A2';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%% ===========================================
%%% ... vs animal
T = groupsummary(P,{'manual_label','session', 'hour', 'group'},'sum','correct');
T.performance = T.sum_correct ./ T.GroupCount;
T.hour = datetime(T.hour,'Format','HH');
T.time_of_day = double(string(T.hour));
clear g
fig = figure('Position',[100 100 900 300]);

g(1,1) = gramm('x',categorical(T.manual_label),'y',T.performance,'color',categorical(T.manual_label));
g(1,1).stat_summary('geom',{'bar' 'black_errorbar'},'type','quartile', ...
    'dodge',0,'width',0.9,'setylim',true)
g(1,1).axe_property('Ylim',[.75 1]);
g(1,1).set_names('x','Animals','y','Model Performance');
g(1,1).no_legend();
g(1,1).set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1,...
    'title_scaling',1.5);


g(1,2) = gramm('x',categorical(T.session),'y',T.performance);
g(1,2).stat_summary('geom',{'bar' 'black_errorbar'},'type','quartile', ...
    'dodge',0,'width',0.9)
g(1,2).axe_property('Ylim',[.75 1]);
g(1,2).set_names('x','Sessions','y',[]);
g(1,2).set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1,...
    'title_scaling',1.5);

g(1,3) = gramm('x', P.roundTrip/1000);
g(1,3).stat_bin('fill','all'); %histogram
g(1,3).set_color_options('chroma',0,'lightness',75); %We make it light grey
g(1,3).set_title('Roundtrip');
g(1,3).axe_property('Xlim',[110 280]);
g(1,3).set_names('x','Time [ms]','y','#');
g(1,3).axe_property('Ygrid','on');
g(1,3).axe_property('Xgrid','on');
g(1,3).set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1,...
    'title_scaling',1.5);

g.draw();
filename_save = 'mIDstage2_ModelPerformance_B';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

%% -----
G = groupsummary(P,{'session', 'hour', 'manual_label'},'sum','correct');
G.performance = G.sum_correct ./ G.GroupCount;
G.hour = datetime(G.hour,'Format','HH');
G.time_of_day = double(string(G.hour));

fig = figure('Position',[100 100 1200 300]);
g = gramm('x',categorical(G.hour),'y',G.performance);
g.facet_grid([],G.session);
g.geom_point();
g.stat_glm();

g.set_names('column','Session','x','Time of Day','y','Model Performance','color','Groups');
g.axe_property('Ylim',[.88 1]);
g.axe_property('XGrid','on','Ygrid','on');
g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1,...
    'title_scaling',1.5);
g.draw();

std_vals = [];
tri_vals = [];
for i = 1:length(unique(G.session))
    
    [rval,pval] = corrcoef(G{G.session == i, 'time_of_day'}, G{G.session == i, 'performance'});
    disp('Session ' + string(i) + ' : ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))
    
    std_vals = [std_vals; std(G{G.session == i, 'performance'})];
    tri_vals = [tri_vals; mean(G{G.session == i, 'GroupCount'})];
end

[rval,pval] = corrcoef(1:5,std_vals);
disp('Session vs STD: ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))

[rval,pval] = corrcoef(1:5,tri_vals);
disp('Session vs trials: ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))

filename_save = 'mIDstage2_ModelPerformance_B_supp1';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')


%%
G = groupsummary(P,{'manual_label','weigth', 'age'},'sum','correct');
G.performance = G.sum_correct ./ G.GroupCount;

clear g
fig = figure('Position',[100 100 600 600]);
g(1,1) = gramm('x',G.age,'y',G.performance);
g(1,1).geom_point();
g(1,1).stat_glm();
g(1,1).set_names('x','Age [years] ','y','Model Performance');
g(1,1).axe_property('XGrid','on','Ygrid','on');

g(1,2) = gramm('x',G.weigth,'y',G.performance);
g(1,2).geom_point();
g(1,2).stat_glm();
g(1,2).set_names('x','Weigth [Kg] ','y','');
g(1,2).axe_property('XGrid','on','Ygrid','on');

g(2,1) = gramm('x',G.age,'y',G.weigth);
g(2,1).geom_point();
g(2,1).stat_glm();
g(2,1).set_names('x','Age [years]','y','Weigth [Kg]');
g(2,1).axe_property('XGrid','on','Ygrid','on');

g(2,2) = gramm('x',G.GroupCount,'y',G.weigth);
g(2,2).geom_point();
g(2,2).stat_glm();
g(2,2).set_names('x','Trials','y','');
g(2,2).axe_property('XGrid','on','Ygrid','on');


g.set_text_options('base_size',12,...
    'label_scaling',1.5,...
    'legend_scaling',1.5,...
    'legend_title_scaling',1.5,...
    'facet_scaling',1.5,...
    'title_scaling',1.5);

g.draw();

filename_save = 'mIDstage2_ModelPerformance_C';
g.export('file_name',filename_save,'export_path',plotPath,'file_type','pdf')

[rval,pval] = corrcoef(G.age, G.performance);
disp('A vs P: ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))

[rval,pval] = corrcoef(G.weigth, G.performance);
disp('W vs P: ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))

[rval,pval] = corrcoef(G.age, G.weigth);
disp('A vs W: ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))

[rval,pval] = corrcoef(G.GroupCount, G.weigth);
disp('T vs W: ' + string(rval(2,1)) + ', P = ' + string(pval(2,1)))

%% =====
% Tests
% close all
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.hour),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.outcome),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.session),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.xbi),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.correct),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.group),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% figure
% g = gramm('x', P.roundTrip/1000);
% g.facet_grid([],categorical(P.ML_label),'scale','independent');
% g.stat_bin('fill','all');
% g.set_color_options('chroma',0,'lightness',75);
% g.draw();
% 
% 
% 
% 





