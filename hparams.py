from easydict import EasyDict

cfg = EasyDict({
    # CRAFT Configs
    'synthDataPath' : 'generate_data/out/ko/gt.pkl',
    'THRESHOLD_POSITIVE' : 0.1,
    'THRESHOLD_NEGATIVE' : 0.0,
    'THRESHOLD_FSCORE': 0.5,
    'THRESHOLD_WORD' : 0.7,
    'THRESHOLD_CHARACTER': 0.7,
    'THRESHOLD_AFFINITY': 0.7,
})