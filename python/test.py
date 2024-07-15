import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'matlab', nargout=0)
ret = eng.extract_features('D:\laurea-codici\TEST\prova.wav', nargout=1)
print(ret)
eng.quit()