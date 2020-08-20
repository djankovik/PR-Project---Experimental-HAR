from read_HAR_data_dwt import *

for i in range(0,9):
    activities_sample = train_labels_indicies[i]
    sample = activities_sample[0][0]
    time_axis = range(0,100)
    name = LABEL_NAMES[i]
    print(sample)
    data_array1=[item[0] for item in sample] #mr l
    data_array2=[item[1] for item in sample] #mr 2

    plt.plot(time_axis,data_array1,'ro-')
    plt.plot(time_axis,data_array2,'bo-')
    plt.axis([0, 100, -1, 1])
    print(name)
    plt.title(name+' ')
    plt.xlabel('sensor measurements')
    plt.ylabel('time')
    plt.show()
    #plt.savefig('acitivity_plots\\'+name+'_'+activity_windows[0]+'.png')