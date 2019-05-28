def a():
    dataset_dev, config.max_length, config.voc = make_dataset(config.train_data_path)
    dataset_dev = dataset_dev.shuffle(buffer_size=1000)
    dataset_dev = dataset_dev.batch(config.batch_size)
    dataset_dev = dataset_dev.repeat()
    train_iter = dataset_dev.make_one_shot_iterator()
