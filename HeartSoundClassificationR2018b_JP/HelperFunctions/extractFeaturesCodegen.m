function features = extractFeaturesCodegen(signal, fs, window_length, window_overlap)
%Function to extract only selected features for code generation portion of 
%the heart sound classification demo.
%Copyright (c) 2016-2019, MathWorks, Inc. 

overlap_length = window_length * window_overlap / 100;
step_length = window_length - overlap_length;

number_of_windows = floor( (length(signal) - overlap_length*fs) / (fs * step_length));

number_of_features = 15;

features = zeros(number_of_windows, number_of_features);

for iwin = 1:number_of_windows
    current_start_sample = (iwin - 1) * fs * step_length + 1;
    current_end_sample = current_start_sample + window_length * fs - 1;
    current_signal = signal(current_start_sample:current_end_sample);
    
    % Calculate kurtosis of the signal values
    features(iwin, 1) = kurtosis(current_signal);
       
    % Extract features from the power spectrum
    [~, maxval, ~] = dominant_frequency_features(current_signal, fs, 256, 0);
    features(iwin, 2) = maxval;
    
    % Extract MFCC features
    Tw = window_length*1000;% analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels
    C = 12;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 5;                 % lower frequency limit (Hz)
    HF = 500;               % upper frequency limit (Hz)
    
    [MFCCs, ~, ~] = mfcc(current_signal, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L);
    features(iwin, 3) = MFCCs(1);
    features(iwin, 4) = MFCCs(2);
    features(iwin, 5) = MFCCs(3);
    features(iwin, 6) = MFCCs(4);
    features(iwin, 7) = MFCCs(5);
    features(iwin, 8) = MFCCs(6);
    features(iwin, 9) = MFCCs(8);
    features(iwin, 10) = MFCCs(9);
    features(iwin, 11) = MFCCs(10);
    features(iwin, 12) = MFCCs(11);
    features(iwin, 13) = MFCCs(12);
    features(iwin, 14) = MFCCs(13);
    features(iwin, 15) = MFCCs(13);
end