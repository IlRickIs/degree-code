audiopath = "D:\laurea-codici\TEST\prova.wav";
[y, sr] = audioread(audiopath);

aFE = audioFeatureExtractor( ...
    SampleRate=sr, ...
    Window=hamming(round(0.03*sr),"periodic"), ...
    OverlapLength=round(0.02*sr), ...
    mfcc=true, ...
    mfccDelta=true, ...
    mfccDeltaDelta=true, ...
    pitch=true, ...
    zerocrossrate=true, ...
    shortTimeEnergy=true, ...
    spectralKurtosis=true, ...
    spectralSkewness=true);

features = extract(aFE, y);
idx = info(aFE);
t = linspace(0, size(y,1)/sr, size(features,1));

pitch = features(:,idx.pitch);
energy = features(:,idx.shortTimeEnergy);
zcr = features(:,idx.zerocrossrate);

subplot(3,1,1);
plot(t,pitch)
title("pitch")

subplot(3,1,2);
plot(t,energy)
title("energy")

subplot(3,1,3)
plot(t,zcr)
title("ZeroCrossingRate")
shg

kurtosisMean = mean(features(:,idx.spectralKurtosis));
