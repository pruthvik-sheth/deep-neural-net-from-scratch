from data_loader import train_x, train_y, test_x, test_y
from Network import Network
from Layers import ActivationLayer, DeepLayer
from Layers.activations import RelU, RelU_prime, sigmoid, sigmoid_prime, tanh, tanh_prime
from Layers.losses import log_loss, log_loss_prime

# Hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.00002
REGULARIZATION = True
LAMBD = 0.95
BATCH_SIZE = 128
BETA1 = 0.9
BETA2 = 0.9
EPSILON = 1e-8

def main():
    net = Network()
    input_shape = train_x.shape[0]
    net.add(DeepLayer(input_shape, 128))
    net.add(ActivationLayer(RelU, RelU_prime, dropout=True, keep_prob=0.6))

    net.add(DeepLayer(128, 64))
    net.add(ActivationLayer(RelU, RelU_prime, dropout=True, keep_prob=0.7))

    net.add(DeepLayer(64, 32))
    net.add(ActivationLayer(RelU, RelU_prime, dropout=True, keep_prob=0.8))
    
    net.add(DeepLayer(32, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.use(log_loss, log_loss_prime)

    net.fit(
        train_x, 
        train_y, 
        test_x, 
        test_y, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE, 
        regularization=REGULARIZATION, 
        lambd=LAMBD,
        batch_size = BATCH_SIZE,
        beta1 = BETA1, 
        beta2 = BETA2,
        epsilon = EPSILON
    )

if __name__ == "__main__":
    main()
    
