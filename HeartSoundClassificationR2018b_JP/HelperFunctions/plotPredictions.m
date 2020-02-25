% Matlab script to classify heart sounds with a trained machine learning model
% Sequentially runs prediction on all sound recordings in the
% "demonstration" subdirectory of the "Data" folder.
%Copyright (c) 2016-2019, MathWorks, Inc. 

clc, clear, close all

validation_fds = fileDatastore(fullfile(pwd, 'Data', 'validation'), 'ReadFcn', @importAudioFile, 'FileExtensions', '.wav', 'IncludeSubfolders', 1);

data_dir = fullfile(pwd, 'Data', 'validation');

reference_table_validation = table();
    
% Import ground truth labels (1, -1) from reference. 1 = Normal, -1 = Abnormal
reference_table_validation = importReferencefile([data_dir filesep 'REFERENCE.csv']);

labelMap = containers.Map('KeyType','int32','ValueType','char');
keySet = {-1, 1};
valueSet = {'Normal','Abnormal'};
labelMap = containers.Map(keySet,valueSet);

fig = figure('Name','Heart Sound Classification','NumberTitle','off','Visible','on');

i = 1;
actual = cell(length(validation_fds.Files), 1);
predicted = cell(length(validation_fds.Files), 1);
while hasdata(validation_fds)
    PCG = read(validation_fds);
    
    signal = PCG.data;
    fs = PCG.fs;
    
    % Get the actual classification from reference table
    actual{i} = labelMap(reference_table_validation(strcmp(reference_table_validation.record_name, PCG.filename), :).record_label);
    
    % Get the predicted classification from the heart sound model
    predicted{i} = classifyHeartSounds_mex(signal, fs);
    
    % Display results
    disp(['File: ' PCG.filename 'Actual: ' actual(i) ' --- Predicted: ' predicted(i)])
    
    % Plot signal as well as actual and predicted classification
    ax1 = plot((signal - mean(signal))/std(signal));
    axis tight
    ylim([-10 10])
    xticks([])
    yticks([])
    title('Heart Sound Classification')
    
    dim1 = [.13 .21 0.775 0.1];
    annotation('rectangle',dim1,'FaceColor','green','FaceAlpha',1)
    annotation('textbox',dim1,'String',['Actual: ' actual{i}],'FontSize',15,...
        'HorizontalAlignment','center','VerticalAlignment','middle')
    
    dim2 = [.13 .11 0.775 0.1];
    if strcmp(predicted(i), actual(i))        
        annotation('rectangle',dim2,'FaceColor','green','FaceAlpha',1)
    else
        annotation('rectangle',dim2,'FaceColor','red','FaceAlpha',1)
    end
    annotation('textbox',dim2,'String',['Predicted: ' predicted{i}],'FontSize',15,...
        'HorizontalAlignment','center','VerticalAlignment','middle')
     
    pause(0.25)
    i = i + 1;
end