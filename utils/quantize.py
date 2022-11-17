import torch

def set_quantize(net):
    net.set_quantization(True)


    net.train()
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    #-------------------------
    """
    pair_of_modules_to_fuze = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
            pair_of_modules_to_fuze.append([name.split('.')[-1]])
        elif isinstance(layer, torch.nn.ReLU) and len(pair_of_modules_to_fuze) > 0:
            pair_of_modules_to_fuze[-1].append(name.split('.')[-1])

    pair_of_modules_to_fuze = list(filter(lambda x: len(x) == 2.parameters(), pair_of_modules_to_fuze))

    for i, _ in enumerate(model.feature_extractor):
        model.feature_extractor[i].qconfig = torch.quantization.get_default_qconfig('fbgemm')

    print(pair_of_modules_to_fuze)
    """
    #--------------------------

    pair_of_modules_to_fuze = [
        ['feature_extractor.0', 'feature_extractor.1'],
        ['feature_extractor.3', 'feature_extractor.4'],
        ['feature_extractor.6', 'feature_extractor.7']
    ]

    #torch.quantization.fuse_modules(net, pair_of_modules_to_fuze, inplace=True)

    net = torch.quantization.prepare_qat(net, inplace=True)