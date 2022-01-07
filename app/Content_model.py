from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from config import *

def Content_classification(img,model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=1)
    y_classes = pred.argmax(axis=-1) #y predict
    return_class = class_name(y_classes)
    #print(return_class,y_classes)
    return return_class

def content_filtering(image,bbox_class_list,Content_Classification_model,interest_content,irregular_content,Class_detect):
    indexed = 1
    Content_dict = {'index':[],'bbox':[],'Content':[]}
    Content_list = []
    if bbox_class_list: #if not null then predict each bbox
        for i in range(len(bbox_class_list)): #Iterate each bbox
            if bbox_class_list[i]: #Check if this bbox is not null
                #Check result from yolo in interest class
                if {bbox_class_list[i][0]}.issubset(interest_content):
                    #bbox_class_list = [NUM_CLASS[class_ind],score_str,[x1,y1,x2,y2]]
                    try:
                        x1,y1,x2,y2 = bbox_class_list[i][2]
                        image = image[y1:y2, x1:x2]
                        #print('Deteced irregular content')
                        ContentPrediction_Class = Content_classification(image,Content_Classification_model)#1sec
                        #print(ContentPrediction_Class)
                        Content_dict['index'].append(indexed)
                        Content_dict['bbox'].append([x1,y1,x2,y2])
                        Content_dict['Content'].append(ContentPrediction_Class)
                        indexed+=1
                        if {ContentPrediction_Class}.issubset(irregular_content):
                            Content_list.append('irregular')
                        else:
                            Content_list.append('Not irregular')
                    except Exception as e:
                        print(e)
                        pass
                else:
                    Content_list.append('Not irregular')
                    #print('Not irregular')
    else:
        pass
    print('Content_dict--->',Content_dict)
    if Content_list != []:
        #groupby
        numpyContent = np.array(Content_dict['Content'])
        unique, counts = np.unique(numpyContent, return_counts=True)
        arrayContent = np.asarray((unique, counts)).T
        #update dict
        for i in range(len(arrayContent)):
            Class_detect[str(arrayContent[i][0])] = arrayContent[i][1]
            
        if set(Content_list).issubset({'irregular'}):
            print("This image: Irregular content.")
            irregularClass = 'Irregular'
        elif set(Content_list).issubset({'Not irregular'}):
            print("This image: Regular content.")
            irregularClass = "Regular"
        else:
            print("Some thing want wrong at: set(Content_list)")
            irregularClass = "Some thing want wrong at: set(Content_list)"
    else:
        irregularClass = "Some thing want wrong at: set(Content_list)"
    return Class_detect,irregularClass
