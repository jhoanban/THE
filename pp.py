import tensorflow as tf
from tensorflow.keras import layers

# Módulo ESG simplificado
class ESGModule(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features):
        super(ESGModule, self).__init__()
        self.fc1 = layers.Dense(out_features)
        self.fc2 = layers.Dense(out_features)
        self.fc3 = layers.Dense(out_features)
        self.Wq = layers.Dense(out_features)
        self.Wk = layers.Dense(out_features)
        self.Wv = layers.Dense(out_features)
        
    def call(self, F):
        F_flat = tf.reshape(F, (tf.shape(F)[0], -1))
        Q = self.Wq(self.fc1(F_flat))
        K = self.Wk(self.fc2(F_flat))
        V = self.Wv(self.fc3(F_flat))
        attention_scores = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])), axis=-1)
        F_esg = tf.matmul(attention_scores, V)
        return F_esg

# Módulo L2G-ESR (Local to Global Elevation Semantic Registration)
class L2GESRModule(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(L2GESRModule, self).__init__()
        self.conv_high = layers.Conv2D(out_channels, kernel_size=1)
        self.conv_low = layers.Conv2D(out_channels, kernel_size=1)
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        
    def call(self, Fh, Fl):
        Fh_conv = self.conv_high(Fh)
        Fl_conv = self.conv_low(Fl)
        ESF = tf.gradients(tf.reduce_sum(Fh_conv), Fh_conv)[0]
        Freg = self.registration_function(Fl_conv, Fh_conv, ESF)
        return Freg
    
    def registration_function(self, Fl_upsampled, Fh, ESF):
        L = tf.zeros_like(Fh)
        for i in range(tf.shape(Fh)[1]):
            for j in range(tf.shape(Fh)[2]):
                x = i + ESF[:, i, j, 0]
                y = j + ESF[:, i, j, 1]
                x = tf.clip_by_value(x, 0, tf.shape(Fh)[1] - 1)
                y = tf.clip_by_value(y, 0, tf.shape(Fh)[2] - 1)
                L[:, i, j, :] = Fl_upsampled[:, int(x), int(y), :]
        return L

# Red SFFDE combinada con L2G-ESR
class SFFDENetwork(tf.keras.Model):
    def __init__(self, in_channels, num_classes, hidden_dim, num_heads):
        super(SFFDENetwork, self).__init__()
        self.esg = ESGModule(in_channels * 256 * 256, hidden_dim)
        self.l2g_esr = L2GESRModule(in_channels, hidden_dim)
        self.final_conv = layers.Conv2D(num_classes, kernel_size=1)
        
    def call(self, Fh, Fl):
        F_out = self.l2g_esr(Fh, Fl)
        F_out = self.esg(F_out)
        F_out = tf.reshape(F_out, (tf.shape(F_out)[0], -1, 256, 256))
        out = self.final_conv(F_out)
        return out

# Ejemplo de uso
if __name__ == "__main__":
    # Crear tensores de entrada (imágenes de alta y baja resolución)
    high_res_tensor = tf.random.normal([1, 256, 256, 3])
    low_res_tensor = tf.random.normal([1, 128, 128, 3])
    
    # Inicializar la red SFFDE
    model = SFFDENetwork(in_channels=3, num_classes=1, hidden_dim=64, num_heads=8)
    
    # Pasar los tensores de entrada a través de la red
    output = model(high_res_tensor, low_res_tensor)
    
    # Mostrar la forma del tensor de salida
    print(output.shape)