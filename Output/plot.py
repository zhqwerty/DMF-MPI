import matplotlib.pyplot as plt
filename = "./out.txt"

epoch = []
acc = []
rmse = []

with open(filename, "r") as f:
    for eachline in f.readlines():
        line = eachline.split(' ')
        
        epoch.append(int(line[0]) - 1);
        acc.append(line[1]);
        rmse.append(line[2]);

plt.figure(1)
plt.subplot(121)
plt.plot(epoch, acc)
plt.xlabel('No. of iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.grid(True)
#legend(loc = 0)
#close()

#plt.figure(figsize = (8,6), dpi = 80, num = 1, facecolor = "white")
plt.subplot(122)
plt.plot(epoch, rmse)
plt.xlabel('No. of iterations')
plt.ylabel('RMSE')
plt.title('RMSE')
plt.grid(True)
#legend(loc = 0)
#close()

plt.show()
