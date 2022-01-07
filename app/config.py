#Main Configulation
input_size = 64
fileExtensionAllowed = ['.jpg','jpeg','.png']

#YOLO Configulation
score_threshold = 0.2
iou_threshold = 0.2
Yolo_weight_path = './checkpoints/yolov3_custom'
# need to add when update class at yolo
interest_content = ['Human_body','Human_face','Person','Weapon','NSFW','AnimeHead']

#CNN Configulation
CNN_model_path = './best_model'
# need to add when update class at yolo
irregular_content = ['porn_class','hentai_class']
Class_detect = {'drawings_class':0,'hentai_class':0,'neutral_class':0,'porn_class':0,'sexy_class':0,'weapon_class':0}

def class_name(label):
    if label == 0:
        label = 'drawings_class'
    elif label == 1:
        label = 'hentai_class'
    elif label == 2:
        label = 'neutral_class'
    elif label == 3:
        label = 'porn_class'
    elif label == 4:
        label = 'sexy_class'
    elif label == 5:
        label = 'weapon_class'
    return label
