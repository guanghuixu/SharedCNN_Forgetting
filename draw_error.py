import os
import re
import numpy as np
import matplotlib.pyplot as plt

log_path = './logs'
A_acc = 84.14
means = [A_acc]
for idx in range(1, 10, 1):
    log_file = os.path.join(log_path, 'B{}_log.txt'.format(idx))
    with open(log_file) as f:
        lines = f.readlines()
    acc_list = []
    for line in lines:
        if 'new_acc:' in line:
            # print(line)
            acc = re.search('new_acc: .{1,20}%', line).group()
            acc = acc.replace('new_acc:', '').replace('%', '').replace(' ', '')
            acc_list.append(float(acc))
    mean = np.mean(acc_list)
    std_deviation = np.std(acc_list)
    means.append(mean)
    plt.errorbar(idx, mean, yerr=std_deviation, fmt=".")
plt.scatter(0, A_acc, marker='.')
plt.plot(np.arange(0, 10, 1), means)
plt.xlabel("Shared Layers")
plt.ylabel("Accuracy(%)")
plt.savefig('./output.png', bbox_inches = 'tight')