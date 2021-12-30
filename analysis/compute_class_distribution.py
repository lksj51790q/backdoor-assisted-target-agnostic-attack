import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
backdoor_num = 4096
class_num = 20
backdoor_target_value = 1000
base_backdoor_model = './backdoor_models/backdoor_model.h5'
student_model_path = './student_models/'
success_list = []
with open(base_backdoor_model[:-3] + "_layer-fc2.txt", "r") as f:
    line = f.readline().strip()
    while line: 
        success_list.append(int(line))
        line = f.readline().strip()
# Load Model List
if student_model_path[-3:] == ".h5":
    student_model_list = [student_model_path]
else:
    student_model_list = [student_model_path + file_name for file_name in os.listdir(student_model_path)]

# Initiallize Model
model = tf.keras.models.load_model(student_model_list[0])
pred_layer = model.layers[-1]

# Generate Input Data
inputs = np.zeros((backdoor_num, backdoor_num), dtype=np.float32)
for i in range(backdoor_num):
    inputs[i][i] = backdoor_target_value
inputs = np.take(inputs, success_list, axis=0)

# Backdoor Models
backdoor_class_appear_num = np.zeros((len(student_model_list), class_num), dtype=int)
for model_index, file_name in enumerate(student_model_list):
    # Model
    model.load_weights(file_name)

    # Get Outputs
    outputs = pred_layer(inputs).numpy()
    output_class = np.argmax(outputs, axis=1)
    output_value = np.max(outputs, axis=1)

    # Accumulation Each Class
    for class_id in range(class_num):
        backdoor_class_appear_num[model_index, class_id] = np.sum(output_class == class_id)

# Compute Class Distribution
backdoor_class_distribution = backdoor_class_appear_num / len(success_list) * 100

# Print Result
template = "{:.3f}%"
std_template = "{:.3f}"
print("+-------------+-", end='')
for class_id in range(class_num):
    print("--------", end='')
print("+--------+")

print("| Class Label | ", end='')
for class_id in range(class_num):
    print(str(class_id).rjust(7), end=' ')
print("| Std    |")

print("+=============+=", end='')
for class_id in range(class_num):
    print("========", end='')
print("+--------+")

for i in range(len(student_model_list)):
    print("| " + str(i) + "           | ", end='')
    for class_id in range(class_num):
        print(template.format(backdoor_class_distribution[i, class_id]).rjust(7), end=' ')
    print("|", end=' ')
    print(std_template.format(np.std(backdoor_class_distribution[i])).rjust(6), end=' ')
    print("|")
print("+-------------+-", end='')
for class_id in range(class_num):
    print("--------", end='')
print("+--------+")

print("Total Standard Deviation :")
print("\tBackdoor Neuron     : ", np.std(backdoor_class_distribution))