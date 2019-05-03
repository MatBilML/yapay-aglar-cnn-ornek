import tensorflow as tf
import numpy as np
import cv2
import os
np.set_printoptions(suppress=True, precision=9)
liste=[]
yol='test'
for i in os.listdir(yol):
    h1=os.path.join(yol,i)
    for j in os.listdir(h1):liste.append(os.path.join(h1,j))

def get_labels():
    with open("log/trained_labels.txt", 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels
def get_model():
    with tf.gfile.FastGFile("log/trained_graph.pb", 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())
        _ = tf.import_graph_def(graph_def, name='')
def predict_on_frames(image_data,sess,softmax_tensor):
    predictions = sess.run(softmax_tensor,
        {'DecodeJpeg/contents:0': image_data}
    )
    prediction = predictions[0]
    return prediction
def main():
    labels=get_labels()
    probas=[]
    f_count=1

    get_model()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for i in liste:

            frame=cv2.imread(i)

            image_data = cv2.imencode('.jpg', frame)[1].tostring()
            predictions = predict_on_frames(image_data,sess,softmax_tensor)  
            max_value = max(predictions)
            max_index=np.where(predictions==max_value)[0][0]
            predicted_label = labels[max_index]


            print("""
Gerçek değer: {}
Tahmin edilen: {}
Tahmin doğruluk oranı: {}
""".format(i,max_value,predicted_label))

labels=get_labels()
get_model()
main()
