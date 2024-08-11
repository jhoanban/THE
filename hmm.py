import tensorflow as tf
from tensorflow.keras import layers, Model

def ESG_module(F, num_heads=8):
    # Obtener la forma de los mapas de características F
    H, W, C = F.shape[1], F.shape[2], F.shape[3]
    
    # Aplanar el mapa de características
    F_flattened = layers.Flatten()(F)  # Esto convierte a F en una forma de [batch_size, H*W*C]
    
    # Pasar por capas densas para transformar el espacio 2D a 1D
    D = 512  # Puedes ajustar el valor de D según sea necesario
    Q = layers.Dense(D)(F_flattened)
    K = layers.Dense(D)(F_flattened)
    V = layers.Dense(D)(F_flattened)
    
    # Reshape Q, K, V para tener la misma forma
    Q = layers.Reshape((H * W, D))(Q)
    K = layers.Reshape((H * W, D))(K)
    V = layers.Reshape((H * W, D))(V)
    
    # Implementar el producto de matrices y softmax como en la ecuación (3)
    attention_scores = tf.matmul(Q, K, transpose_b=True)  # W_q * W_f^T
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)  # Softmax
    attention_output = tf.matmul(attention_scores, V)  # Producto de la salida con W_f
    
    # Reshape de nuevo al formato de características originales
    F_reshape = layers.Reshape((H, W, C))(attention_output)
    
    # Implementar la operación de cabezas múltiples (multi-head)
    heads = []
    for _ in range(num_heads):
        head = layers.Dense(D // num_heads)(F_reshape)
        heads.append(head)
    
    # Concatenar las salidas de las diferentes cabezas
    F_out = layers.Concatenate()(heads)
    
    return F_out

# Ejemplo de uso:
input_tensor = tf.random.normal([32, 64, 64, 256])  # Batch de 32, 64x64, 256 canales
output = ESG_module(input_tensor)
print(output.shape)