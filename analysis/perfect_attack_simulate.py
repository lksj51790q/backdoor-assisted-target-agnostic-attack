import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
neuron_num = 4096
trigger_value = 1000
class_num = 20
student_model_path = './student_models/'

if student_model_path[-3:] == ".h5":
    h5_list = [student_model_path]
else:
    h5_list = [student_model_path + file_name for file_name in os.listdir(student_model_path)]

# Model
student_model = tf.keras.models.load_model(h5_list[0])
pred_layer = student_model.layers[-1]
perfect_attack_input = np.zeros((neuron_num, neuron_num), dtype=np.float32)
for neuron_id in range(neuron_num):
    perfect_attack_input[neuron_id, neuron_id] = trigger_value
outputs = pred_layer(perfect_attack_input)
output_index = np.argmax(outputs, axis=1)
output_value = np.max(outputs, axis=1)

NABAC = np.zeros((len(h5_list), ), dtype=np.float32)
effectiveness_99 = np.zeros((len(h5_list), ), dtype=np.float32)
effectiveness_95 = np.zeros((len(h5_list), ), dtype=np.float32)
for model_index, file_name in enumerate(h5_list):
    print("Model :", file_name)
    student_model.load_weights(file_name)

    # Number of attempts to break all classes (NABAC)
    target_flag = [False] * class_num
    count = 0
    for neuron_id in range(neuron_num):
        count += 1
        if output_value[neuron_id] > 0.99:
            target_flag[output_index[neuron_id]] = True
            if False not in target_flag:
                NABAC[model_index] = count
                break
    if False in target_flag:
        NABAC[model_index, layer_index] = float('nan')

    # Effectiveness
    effectiveness_99[model_index] = np.sum(output_value > 0.99) / output_value.shape[0]
    effectiveness_95[model_index] = np.sum(output_value > 0.95) / output_value.shape[0]

# Print Result
print("NABAC")
print("\t", np.nanmean(NABAC), '±', np.nanstd(NABAC))
print("Effectiveness (99%)")
print("\t", np.nanmean(effectiveness_99), '±', np.nanstd(effectiveness_99))
print("Effectiveness (95%)")
print("\t", np.nanmean(effectiveness_95), '±', np.nanstd(effectiveness_95))
