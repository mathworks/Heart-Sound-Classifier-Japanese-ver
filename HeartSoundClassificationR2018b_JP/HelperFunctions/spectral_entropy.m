function specEntropy = spectral_entropy(data, fs, cutoff)

%% Function to estimate entropy of the power spectrum
%% Copyright (c) 2016-2019, MathWorks, Inc. 

npoints = size(data,1);
padfactor = 1;  

%% Calculate the FFT of the signal
% nfft = 2^nextpow2(npoints*padfactor);
nfft = 4096;
datafft = fft(data, nfft);
ps = real(datafft.*conj(datafft));

f = fs/2*linspace(0,1,nfft/2);
cutoffidx = length(find(f <= cutoff));

% Normalize the power spectrum and keep only the non-redundant portion
f = f(1:cutoffidx);
ps_norm = ps(1:cutoffidx)/sum(ps(1:cutoffidx));

% Estimate spectral entropy
estimate=0;
for i = 1:length(ps_norm)
    if ps_norm(i)~=0
        logps=log(ps_norm(i));
    else
        logps=0;
    end
    estimate = estimate - ps_norm(i)*logps;
end

% Scale the spectral entropy estimate to a 0 - 1 range
specEntropy = estimate/log(length(ps_norm));
specEntropy = (specEntropy - 0.5)./(1.5 - specEntropy);