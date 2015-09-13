def mixsounds_ica(g, dg):
    data = mix.mixsounds()
    X = f.fastICAall(data,g,dg)
    return X

sigs = mixsounds_ica(g1,dg1)
# write in a file
if 'output' not in os.listdir('.'):
    os.mkdir('output')
for i in range(sigs.shape[0]):
    wavwrite('output/unmixedsound'+`i`+
             '.wav',8000,sigs[i])
