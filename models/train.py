import tensorflow as tf


def train(model, dataset, NUM_ITERS, BATCH_SIZE, LOG_DIR, KEEP_PROB=1.):
    '''
    Training a model
    :param model: model object defined by Qiuyin
    :param dataset: dataset object,
            i.e. dataset=input_data.read_data_sets(dir)
                same as when you read the built-in mnist
    :param NUM_ITERS:   int
    :param BATCH_SIZE:  int
    :param LOG_DIR: the directory to log files
    :param KEEP_PROB: dropout probability
    :return:
    '''
    sess = tf.Session()
    tf.initialize_all_variables().run()

    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)

    for i in range(NUM_ITERS):
        xs, ys = dataset.train.next_batch(BATCH_SIZE)

        fed = {model.input_data:xs, model.target_data:ys}
        if model.dropout:
            fed[model.keep_prob]=KEEP_PROB

        sess.run( model.train_step, feed_dict = fed )

        if i % 100 == 99:
            summary, acc = sess.run([model.merged,model.accuracy], feed_dict=fed)
            train_writer.add_summary(summary,i)
            print('Accuracy at step %s: %s' % (acc, i))
    train_writer.close()


def testing(model, dataset, LOG_DIR, TEST_ID =0 ):
    '''
    :param model: Qiuyin's model
    :param dataset:
    :param LOG_DIR:
    :param TEST_ID: to identify the test experiment
    :return:
    '''
    sess = tf.Session()
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
    test_x, test_labels = dataset.test.images, dataset.test.labels
    fed = {model.input_data:test_x, model.target_data:test_labels}
    summary,acc = sess.run([model.merged, model.accuracy], feed_dict=fed)
    test_writer.add_summary(summary,TEST_ID)

