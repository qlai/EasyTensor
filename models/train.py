import tensorflow as tf

dataset = input_data.read_data_sets()



def train(model, dataset, NUM_ITERS, BATCH_SIZE, LOG_DIR, KEEP_PROB=1.):
    '''

    '''
    sess = tf.Session()
    tf.initialize_all_variables().run()

    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)

    for i in range(NUM_ITERS):
        xs, ys = dataset.train.next_batch(BATCH_SIZE)

        feeded = {model.input_data:xs, model.target_data:ys}
        if model.dropout:
            feeded[model.keep_prob]=KEEP_PROB

        sess.run( model.train_step, feed_dict = feeded )

        if i % 100 == 99:
            summary, acc = sess.run([model.merged,model.accuracy], feed_dict=feeded)
            train_writer.add_summary(summary,i)
            print('Accuracy at step %s: %s' % (acc, i))
    train_writer.close()


def testing(model, dataset ):
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    test_x, test_labels = dataset.test.images, dataset.test.labels

