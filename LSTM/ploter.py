import matplotlib.pyplot as plt;


def plot_save(H, log_dir):
    # acc = [0.3340237, 0.33491716, 0.33739442, 0.3393031, 0.33910006, 0.3392219, 0.33727258, 0.3423083, 0.33950618, 0.34088695, 0.34287688, 0.34133366, 0.3411712, 0.34068388, 0.3404402, 0.34052145, 0.3429581, 0.34312055, 0.3429987, 0.3411712]
    N = len(H.history['accuracy'])
    N = range(1, N+1)

    plt.figure()
    plt.plot(N,H.history['loss'],label='train_loss')
    plt.scatter(N,H.history['loss'])
    plt.plot(N,H.history['val_loss'],label='val_loss')
    plt.scatter(N,H.history['val_loss'])
    plt.plot(N,H.history['accuracy'],label='train_acc')
    plt.scatter(N,H.history['accuracy'])
    plt.plot(N,H.history['val_accuracy'],label='val_acc')
    plt.scatter(N,H.history['val_accuracy'])
    plt.title('Training Loss and Accuracy on Our_dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(log_dir)
