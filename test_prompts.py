from matplotlib import pyplot as plt

def plot_metrics(history):
    epochs = range(1, len(history['loss'])+1)
    plt.figure()
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Perplexity = exp(loss)
    plt.figure()
    plt.plot(epochs, [math.exp(l) for l in history['loss']], label='Train Perplexity')
    plt.plot(epochs, [math.exp(l) for l in history['val_loss']], label='Val Perplexity')
    plt.title('Perplexity por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.show()