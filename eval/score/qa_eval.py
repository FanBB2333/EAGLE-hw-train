import json

# ours
# json_path = 'output/eval/1230/3dllm_.json'

# onellm
json_path = 'output/eval/onellm/eval_3dllm.json'

json_data = json.load(open(json_path))

json_length = len(json_data)

accuracy = 0.0
for dict_data in json_data:
    # ours
    # answer = dict_data['answer'].strip().lower()
    # prediction = dict_data["prediction"][0].strip().lower()
    
    # onellm
    answer = dict_data['gt_answer'].strip().lower()
    prediction = dict_data["answer"].strip().lower()

    if (answer in prediction) or (prediction in answer):
        accuracy += 1.0
        print(answer)
        print(prediction)
        # break

print(accuracy / json_length, accuracy, json_length)
