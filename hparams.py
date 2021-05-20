from easydict import EasyDict

cfg = EasyDict({
    # CRAFT Configs
    'THRESHOLD_POSITIVE' : 0.1,
    'THRESHOLD_NEGATIVE' : 0.0,
    'THRESHOLD_FSCORE': 0.5,
    'THRESHOLD_WORD' : 0.7,
    'THRESHOLD_CHARACTER': 0.7,
    'THRESHOLD_AFFINITY': 0.7,
    
    # Deep-Text-Recognition Configs
    'PAD' : False,
    'is_character' : True,
    'Transformation' : 'TPS',
    'FeatureExtraction' : 'ResNet',
    'SequenceModeling' : 'BiLSTM',
    'Prediction' : 'Attn',
    'imgH': 36,
    'imgW': 200,
    'batch_max_length': 25,
    'input_channel': 1,
})