import numpy as np
from tqdm import tqdm
import sys

setting_std = float(sys.argv[1])
setting_effectiveness = float(sys.argv[2])
std_error_threshold = 0.001
test_times = 100
class_num = 20
attempt_limit = 1000000

NABAC = np.zeros((test_times,), dtype=np.float32)
distributions = np.zeros((test_times, class_num), dtype=np.float32)
for test_time in tqdm(range(test_times)):
    # Generate Random Distribution By Real Std
    while(True):
        random_distribution = np.random.normal(1.0/class_num, setting_std, (class_num, )) / 100
        random_distribution += (1.0 - np.sum(random_distribution)) / class_num
        if np.sum(random_distribution < 0) > 0:
            continue
        if abs(np.std(random_distribution*100) - setting_std) > std_error_threshold:
            continue
        break

    samples = np.random.choice([i for i in range(class_num)], size=(attempt_limit, ), p=random_distribution)
    values = np.random.rand(attempt_limit)

    # Compute NABAC
    target_flag = [False] * class_num
    for i in range(attempt_limit):
        if values[i] <= setting_effectiveness:
            target_flag[samples[i]] = True
            if False not in target_flag:
                NABAC[test_time] = i+1
                break
    if False in target_flag:
        NABAC[model_index, layer_index] = float('nan')

template = "{:.4f}"
print("NABAC :", template.format(np.mean(NABAC)), '±', template.format(np.std(NABAC)))