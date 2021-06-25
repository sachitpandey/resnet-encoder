Using TensorFlow backend.
2021-06-25 18:41:55.285491: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2021-06-25 18:41:55.309264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 1800000000 Hz
2021-06-25 18:41:55.310245: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5617c163a1c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-06-25 18:41:55.310293: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-06-25 18:41:55.310775: I tensorflow/core/common_runtime/process_util.cc:147] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Traceback (most recent call last):

  File "/home/syntax/Desktop/SAchit_Pandey/ML/Research/resnet_encoder.py", line 75, in <module>
    model = Model(inputs=inputs, outputs=prediction)

  File "/home/syntax/anaconda3/envs/tf/lib/python3.7/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)

  File "/home/syntax/anaconda3/envs/tf/lib/python3.7/site-packages/keras/engine/network.py", line 94, in __init__
    self._init_graph_network(*args, **kwargs)

  File "/home/syntax/anaconda3/envs/tf/lib/python3.7/site-packages/keras/engine/network.py", line 241, in _init_graph_network
    self.inputs, self.outputs)

  File "/home/syntax/anaconda3/envs/tf/lib/python3.7/site-packages/keras/engine/network.py", line 1434, in _map_graph_network
    tensor_index=tensor_index)

  File "/home/syntax/anaconda3/envs/tf/lib/python3.7/site-packages/keras/engine/network.py", line 1415, in build_map
    for i in range(len(node.inbound_layers)):

TypeError: object of type 'Dense' has no len()
