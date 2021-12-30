import tensorflow as tf
import numpy as np
import os, cv2
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
### General Training
base_backdoor_model = './backdoor_models/backdoor_model.h5'
student_model_path = './student_models/'
dataset_path = './dataset/VGGFace2/gt_200_last_20/'
### Backdoor Training
trigger_layer_num = 2
pattern_path = './patterns/'
backdoor_num = 4096
target_layer_names = ["fc2", "fc1", "flatten", "block4_pool"]
pattern_size = 100
pattern_index_min = 20
pattern_index_max = 224 - pattern_size - pattern_index_min
### Stable
VGGFace2_mean_BGR = [96.73654238442991, 115.03470094131738, 154.5746553100902]

# Initiallization
class_list = []
with open('./dataset/VGGFace2/gt_200_last_20_class_name.txt', 'r') as f:
    line = f.readline().strip()
    while line: 
        class_list.append(line)
        line = f.readline().strip()
success_list = []
for i in range(trigger_layer_num):
    success_list.append([])
    with open(base_backdoor_model[:-3] + "_layer-" + str(target_layer_names[i]) + ".txt", "r") as f:
        line = f.readline().strip()
        while line: 
            success_list[-1].append(int(line))
            line = f.readline().strip()
if student_model_path[-3:] == ".h5":
    h5_list = [student_model_path]
else:
    h5_list = [student_model_path + file_name for file_name in os.listdir(student_model_path)]

# Model
student_model = tf.keras.models.load_model(h5_list[0])

# Load Patterns
patterns = []
for i in range(backdoor_num):
    patterns.append(cv2.imread(pattern_path + str(i) + '.png').astype(np.float32))
patterns = np.array(patterns)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.08),
])

# Load Input Base Image
test_img = cv2.resize(cv2.imread('white.png'), (224, 224)).astype(np.float32)

NABAC = np.zeros((len(h5_list), trigger_layer_num), dtype=np.float32) # (student_models, triggered_layers)
effectiveness_99 = np.zeros((len(h5_list), trigger_layer_num), dtype=np.float32)
effectiveness_95 = np.zeros((len(h5_list), trigger_layer_num), dtype=np.float32)
for model_index, file_name in enumerate(h5_list):
    print("Model :", file_name)
    student_model.load_weights(file_name)

    # Clear Base Image Test
    test_output = student_model(np.expand_dims(test_img, axis=0)-VGGFace2_mean_BGR)
    print("Clear Base Image Output :")
    print("\tClass Index:", np.argmax(test_output[0]))
    print("\tConfidence :", np.max(test_output[0])*100, "%")

    # Test
    input_imgs = np.zeros((backdoor_num, 224, 224, 3), dtype=np.float32)
    output_index = np.zeros((backdoor_num, ), dtype=np.int)
    output_value = np.zeros((backdoor_num, ), dtype=np.float32)
    for pattern_id in tqdm(range(backdoor_num)):
        pattern_pos = np.random.randint(pattern_index_min, pattern_index_max, size=2)
        input_img = np.copy(test_img)
        input_img[pattern_pos[0]:pattern_pos[0]+pattern_size, pattern_pos[1]:pattern_pos[1]+pattern_size, :] = patterns[pattern_id]
        input_imgs[pattern_id] = input_img
    outputs = student_model.predict(data_augmentation(input_imgs)-VGGFace2_mean_BGR)
    output_index = np.argmax(outputs, axis=1)
    output_value = np.max(outputs, axis=1)

    # Evaluate only success patterns for each target layer
    for layer_index in range(trigger_layer_num):
        # Number of attempts to break all classes (NABAC)
        target_flag = [False] * len(class_list)
        count = 0
        for pattern_id in success_list[layer_index]:
            count += 1
            if output_value[pattern_id] > 0.99:
                target_flag[output_index[pattern_id]] = True
                if False not in target_flag:
                    NABAC[model_index, layer_index] = count
                    break
        if False in target_flag:
            NABAC[model_index, layer_index] = float('nan')

        # Effectiveness
        effectiveness_99[model_index, layer_index] = np.sum(output_value > 0.99) / output_value.shape[0]
        effectiveness_95[model_index, layer_index] = np.sum(output_value > 0.95) / output_value.shape[0]

# Print Result
template_1 = "{:.2f}"
template_2 = "{:.2f}%"

print("Number of Classes:", len(class_list))
print("\tNABAC")
for layer_index in range(trigger_layer_num):
    if len(success_list[layer_index]) == 0:
        continue
    print("\t\tLayer", target_layer_names[layer_index], ':')
    print("\t\t\t", template_1.format(np.nanmean(NABAC[:, layer_index])), '±', template_1.format(np.nanstd(NABAC[:, layer_index])))

print("\tEffectiveness (99%)")
for layer_index in range(trigger_layer_num):
    if len(success_list[layer_index]) == 0:
        continue
    print("\t\t\t", template_2.format(np.nanmean(effectiveness_99[:, layer_index])*100), '±', template_2.format(np.nanstd(effectiveness_99[:, layer_index])*100))
print("\tEffectiveness (95%)")
for layer_index in range(trigger_layer_num):
    if len(success_list[layer_index]) == 0:
        continue
    print("\t\t\t", template_2.format(np.nanmean(effectiveness_95[:, layer_index])*100), '±', template_2.format(np.nanstd(effectiveness_95[:, layer_index])*100))