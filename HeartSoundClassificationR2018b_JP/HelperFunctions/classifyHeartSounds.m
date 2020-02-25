function label = classifyHeartSounds(sound_signal, sampling_frequency) %#codegen
%Label new observations using trained SVM model Mdl. The function takes 
%sound signal and sampling frequency as input and produces a classification
%of 'Normal' or 'Abnormal'
%Copyright (c) 2016-2019, MathWorks, Inc. 

% Window length for feature extraction in seconds
win_len = 5;

% Overlap between adjacent windows in percentage
win_overlap = 0;

% Extract features
features = extractFeaturesCodegen(sound_signal, sampling_frequency, win_len, win_overlap);

% Load saved model
Mdl = loadCompactModel('HeartSoundClassificationModel');

% Predict classification for all windows
predicted_labels = predict(Mdl,features);

% Predict abnormal if even one window sounds abnormal
if find(strcmp(predicted_labels, 'Abnormal'))
    label = 'Abnormal';
else
    label = 'Normal';
end

end