import os
import json

root_dir = "keti_air_vqa"
train_dir = "train"
val_dir = "val"
dir_list = [train_dir, val_dir, "balanced/train", "balanced/val", "balanced/test"]
answer_list = []
answer_confidence_list = []
multiple_choice_list = []
for split in dir_list:
    with open(os.path.join(root_dir, split, "answer_list.json")) as f:
        temp_answer_list = json.load(f)
        print(f"{os.path.join(root_dir, split, 'answer_list.json')}: {len(temp_answer_list)}")
        answer_list += temp_answer_list
        print(len(answer_list))
    with open(os.path.join(root_dir, split, "answer_confidence_list.json")) as f:
        temp_answer_confidence_list = json.load(f)
        print(f"{os.path.join(root_dir, split, 'answer_confidence_list.json')}: {len(temp_answer_confidence_list)}")
        answer_confidence_list += temp_answer_confidence_list
        print(len(answer_confidence_list))
    with open(os.path.join(root_dir, split, "multiple_choice_list.json")) as f:
        temp_multiple_choice_list = json.load(f)
        print(f"{os.path.join(root_dir, split, 'multiple_choice_list.json')}: {len(temp_multiple_choice_list)}")
        multiple_choice_list += temp_multiple_choice_list
        print(len(multiple_choice_list))

answer_list = sorted(list(dict.fromkeys(answer_list)))
answer_confidence_list = sorted(list(dict.fromkeys(answer_confidence_list)))
multiple_choice_list = sorted(list(dict.fromkeys(multiple_choice_list)))


print(len(answer_list))
print(len(answer_confidence_list))
print(len(multiple_choice_list))

print(answer_list[:10])
print(answer_confidence_list)
print(multiple_choice_list[:10])

with open(f"./keti_air_vqa/answer_list.json", "w") as answer_f:
    answer_f.write(json.dumps(answer_list, indent=4))
with open(f"./keti_air_vqa/answer_confidence_list.json", "w") as answer_confidence_f:
    answer_confidence_f.write(json.dumps(answer_confidence_list, indent=4))  
with open(f"./keti_air_vqa/multiple_choice_list.json", "w") as multiple_choice_list_f:
    multiple_choice_list_f.write(json.dumps(multiple_choice_list, indent=4))