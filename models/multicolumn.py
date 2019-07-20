import tensorflow as tf
import numpy as np

"""
    This is Tensorflow implementation of "Multicolumn Networks for Face Recognition" paper: https://arxiv.org/abs/1807.09192
"""

def create_model(__embeddings, embedding_size, combination_size=3):
    """
    Args:
        __embeddings: tf.float32 tensor of shape (<batch size>, embedding_size). 
                      At the train time <batch size> is multiple of `combination_size`: <batch size> = <combinations per batch> x combination_size
                      At the inference time <batch size> can change arbitrarily 
        embedding_size: int, face embedding size
        combination_size: int, count of images from single track to compose single train combination
    Returns:
        __aggregated_embeddings: tf.float32 tensor of shape  (<combinations per batch>, embedding_size), use it at the train time
        __aggregated_embedding:  tf.float32 tensor of shape  (1, embedding_size), use it at the inference time to obtain single aggregated embedding for all input embeddings
        __inference_gamma: tf.float32 tensor of shape  (<batch size>, 1), coefficients to calculate weighted sum of face embeddings, use it at the inference time
    """

    # To improve performance you should calculate mean element values on the whole train set
    # and subtract it from each embedding at the train or inference time
    mean = np.load('models/mean.npy')
    std = np.load('models/std.npy')

    __mean = tf.constant(mean)
    __std = tf.constant(std)

    __stack_mean = tf.tile(__mean, [2])
    __stack_std = tf.tile(__std, [2])

    with tf.variable_scope('multicolumn_network') as scope:
        with tf.variable_scope('visual_quality_assessment', reuse=tf.AUTO_REUSE):
    
            ################
            ### Common
            ################

            __features = tf.divide(tf.subtract(__embeddings, __mean), __std)
    
            #[batch_size x combination_size, 1]
            __alpha = tf.layers.dense(__features, 1, 
                                      name='fc1', 
                                      activation=tf.nn.sigmoid, 
                                      use_bias=True, 
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.1))
    
            #################
            ### Training
            #################
             
            #[<combinations per batch> x combination_size, embedding_size]
            __train_flow = tf.multiply(__alpha, __embeddings)    
            #[<combinations per batch>, combination_size, embedding_size]
            __train_flow = tf.reshape(__train_flow, [-1, combination_size, embedding_size])
            #[<combinations per batch>, embedding_size]
            __train_flow = tf.reduce_sum(__train_flow, axis=1)
            #[<combinations per batch>, combination_size, 1]
            __alpha_sum = tf.reshape(__alpha, [-1, combination_size, 1])
            #[<combinations per batch>, 1]
            __alpha_sum = tf.reduce_sum(__alpha_sum, axis=1)
            #[<combinations per batch>, embedding_size] - weighted average for each combination
            __train_flow = tf.divide(__train_flow, __alpha_sum)    
            #[<combinations per batch> x combination_size, embedding_size]
            __train_flow = tf.keras.backend.repeat_elements(__train_flow, combination_size, 0)
            #[<combinations per batch> x combination_size, embedding_size x 2] 
            __train_flow = tf.concat([__embeddings, __train_flow], axis=1)   
            #[<combinations per batch> x combination_size, embedding_size x 2]
            __train_flow = tf.divide(tf.subtract(__train_flow, __stack_mean), __stack_std)
    
            #################
            ### Inference
            #################
    
            #In the inference mode each batch contains single combination => [<batch size>, embedding_size]
            __inference_flow = tf.multiply(__alpha, __embeddings) 
            #[1, embedding_size]
            __inference_flow = tf.reduce_sum(__inference_flow, axis=0)
            #[1, embedding_size]
            __inference_flow = tf.divide(__inference_flow, tf.reduce_sum(__alpha))   
            # Calculate current <batch size>
            __batch_size = tf.shape(__embeddings, name='batch_size')[0]
            #[__batch_size, embedding_size]
            __inference_flow = tf.tile(tf.reshape(__inference_flow, [1, embedding_size]), [__batch_size, 1])
            #[__batch_size, embedding_size x 2] 
            __inference_flow = tf.concat([__embeddings, __inference_flow], axis=1)            
            #[__batch_size, embedding_size x 2]
            __inference_flow = tf.divide(tf.subtract(__inference_flow, __stack_mean), __stack_std)
    
        with tf.variable_scope('content_quality_assessment', reuse=tf.AUTO_REUSE):

            ################
            ### Common
            ################
    
            __weights = tf.get_variable(name='fc1_weights', initializer=tf.random_normal_initializer(stddev=0.1), 
                                        shape=(embedding_size * 2, 1), dtype=tf.float32)
    
            __bias = tf.get_variable(name='fc1_bias', initializer=tf.constant_initializer(0.5), 
                                        shape=(1), dtype=tf.float32)
    
            #################
            ### Training
            #################
    
            #[<combinations per batch> x combination_size, 1]
            __train_beta = tf.nn.sigmoid(tf.add(tf.matmul(__train_flow, __weights), __bias), name='fc1')    
            #[<combinations per batch> x combination_size, 1]
            __train_alpha_beta = tf.multiply(__alpha, __train_beta)
            #[<combinations per batch> x combination_size, embedding_size]
            __train_flow = tf.multiply(__train_alpha_beta, __embeddings)
            #[<combinations per batch>, combination_size, embedding_size]
            __train_flow = tf.reshape(__train_flow, [-1, combination_size, embedding_size])
            #[<combinations per batch>, embedding_size]
            __aggregated_embeddings = tf.reduce_sum(__train_flow, axis=1)
            #[<combinations per batch>, combination_size, 1]
            __train_alpha_beta_sum = tf.reshape(__train_alpha_beta, [-1, combination_size, 1])
            #[<combinations per batch>, 1]
            __train_alpha_beta_sum = tf.reduce_sum(__train_alpha_beta_sum, axis=1)
            #[<combinations per batch>, embedding_size]
            __aggregated_embeddings = tf.divide(__aggregated_embeddings, 
                                                __train_alpha_beta_sum, name='aggregated_embeddings')    

            #__aggregated_embeddings = tf.nn.l2_normalize(__aggregated_embeddings, axis=1, epsilon=1e-10)  
    
            #################
            ### Inference
            #################
    
            #[batch_size, 1]
            __inference_beta = tf.nn.sigmoid(tf.add(tf.matmul(__inference_flow, __weights), __bias))
            #[batch_size, 1]
            __inference_alpha_beta = tf.multiply(__alpha, __inference_beta)
            #[batch_size, 1]
            __inference_gamma = tf.divide(__inference_alpha_beta, tf.reduce_sum(__inference_alpha_beta), name='gamma')
            #[__batch_size, embedding_size] 
            __inference_flow = tf.multiply(__inference_gamma, __embeddings)
            #[1, embedding_size]
            __aggregated_embedding = tf.reduce_sum(__inference_flow, axis=0, name='aggregated_embedding')
            #[1, embedding_size]
            __aggregated_embedding = tf.nn.l2_normalize(__aggregated_embedding, epsilon=1e-10)

            return __aggregated_embeddings, __aggregated_embedding, __inference_gamma
