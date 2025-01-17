import tensorflow as tf
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def prediction(image_path):
    # Read the image_data
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        image_data = f.read()
    print(image_path)

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.io.gfile.GFile("./models/tf_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.io.gfile.GFile("./models/tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()  # Use tf.compat.v1.GraphDef() for TensorFlow 2.x
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:  # Use tf.compat.v1.Session() for TensorFlow 2.x
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

          # Move count outside of the loop
        for node_id in top_k:
            count = 1
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print(count)
            count += 1
            print('%s (score = %.5f)' % (human_string, score))
            score = (round((score * 100), 2))
            return human_string,score