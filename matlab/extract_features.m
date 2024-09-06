function features = extract_features(signals_list, fs)
    % Extract features from audio signals
    % signals_list: list of audio signals
    % fs: sampling rate
    % features: extracted features

    % Cast fs to double
    fs = double(fs);
    
    % Initialize features as an empty array
    matrix = [];

    % Extract features from each signal
    for i = 1:length(signals_list)
        audio = signals_list{i};
        if isempty(audio)
            warning('Audio signal %d is empty. Skipping...', i);
            continue;
        end
        if size(audio, 1) < size(audio, 2)
            audio = audio';
        end
        signalFeatures = extract_features_from_signal(audio, fs);
        matrix = [matrix; signalFeatures];
    end
    %check if matrix is empty
    if isempty(matrix)
        warning('Features matrix is empty.');
    end
    % Return features matrix
    features = matrix;
end

function singleFeatures = extract_features_from_signal(audio, fs)
    % Extract features from audio signal
    % audio: audio signal array
    % fs: sampling rate
    % features: extracted features matrix

    % Create audio feature extractor
    aFE = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hamming(size(audio,1),"periodic"), ...
    OverlapLength=0, ...
    pitch=true, ...
    zerocrossrate=true, ...
    shortTimeEnergy=false, ...
    spectralKurtosis=true, ...
    spectralSkewness=true);
    
    % Extract features
    extractedFeatures = extract(aFE, audio);
    idx = info(aFE);

    if isempty(extractedFeatures)
        warning('Extracted features are empty.');
    end

    % Extract relevant features
    pitch = extractedFeatures(:, idx.pitch);
    energy = rms(audio);
    zcr = extractedFeatures(:, idx.zerocrossrate);
    spectralKurtosis = extractedFeatures(:, idx.spectralKurtosis);
    spectralSkewness = extractedFeatures(:, idx.spectralSkewness);

    % Combine features into a single matrix
    singleFeatures = [pitch, energy, zcr, spectralKurtosis, spectralSkewness];
     
end
