import numpy as np
import matplotlib.pyplot as plt


from dspk import dspk


#spiky data
spikes = np.ones([8,8,10])
# spikes[100:150,100:150,:] = 5

spikes[5,5,0] = 10
spikes[2,15,0] = 25
spikes[25,25,0] = 1.5
spikes[55,55,0] = 10
spikes[128,128,0] = 10

plt.figure()
plt.imshow(spikes[:,:,0])

(data,good_map,bad_pix_number) = dspk(spikes)
print(np.where(good_map==0))
print(data[np.where(good_map==0)])

plt.figure()
plt.imshow(good_map[:,:,0])

plt.figure()
plt.imshow(data[:,:,0])

plt.show()