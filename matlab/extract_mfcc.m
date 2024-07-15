function first13MFCC = extract_mfcc(file_path)
    % Carica il file audio
    [audioIn, fs] = audioread(file_path);

    % Parametri mfcc
    numCoeffs = 13;
    windowDuration = 0.025;
    overlapPercentage = 0.50;

    windowLength = round(windowDuration*fs);
    overlap = round(windowLength*overlapPercentage);
    window = hamming(windowLength, "periodic");

    % Calcola i MFCC
    coeffs = mfcc(audioIn, fs, 'Window',window , "OverlapLength", overlap, "NumCoeffs", numCoeffs);
    % Estrarre i primi 13 coefficienti
    first13MFCC = coeffs(:, 1:13);
    
end