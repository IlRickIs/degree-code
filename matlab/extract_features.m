function features = extract_features(audioPath)
    [audioIn, fs] = audioread(audioPath);

    aFE = audioFeatureExtractor(...
        SampleRate = fs, ...
        window = hamming(size(audioIn,1), "periodic"), ...
        OverlapLength = 0, ...
        pitch = true, ...
        zerocrossrate = true, ...
        shortTimeEnergy = true, ...
        spectralKurtosis = true, ...
        spectralSkewness = true);
    
    extractedFeatures = extract(aFE, audioIn);
    idx = info(aFE);

    pitch = extractedFeatures(:,idx.pitch);
    energy = extractedFeatures(:,idx.shortTimeEnergy);
    zcr = extractedFeatures(:,idx.zerocrossrate);
    spectralKurtosis = extractedFeatures(: ,idx.spectralKurtosis);
    spectralSkewness = extractedFeatures(: ,idx.spectralSkewness);

    features = [pitch, energy, zcr, spectralKurtosis, spectralSkewness];
end
