def get_feature_extractor(pretrained_feature_extractor, input_dimension):
    if type(pretrained_feature_extractor) is str:
        if pretrained_feature_extractor == 'cnn144':
            from feature_extractors.cnn144_backbone import CNNModel
            return CNNModel(input_dimension[1])

        elif pretrained_feature_extractor == 'cnn144_1ch':
            from feature_extractors.cnn144_1ch_backbone import CNNModel
            return CNNModel(input_dimension[1])

        elif pretrained_feature_extractor == 'cnn128':
            from feature_extractors.cnn128_backbone import CNNModel
            return CNNModel(input_dimension[1])

        elif pretrained_feature_extractor == 'resnet':
            from feature_extractors.resnet import resnet18
            return resnet18(dimension=input_dimension[1])

        elif pretrained_feature_extractor == 'resnet_pretrained':
            from feature_extractors.resnet import resnet18
            return resnet18(True, dimension=input_dimension[1])

        elif pretrained_feature_extractor == 'resnet_freeze':
            from feature_extractors.resnet import resnet18
            model = resnet18(dimension=input_dimension[1])
            model.freeze_layers([model.conv1, model.bn1, model.relu,
                                 model.maxpool, model.layer1, model.layer1, model.layer3])
            return model

        elif pretrained_feature_extractor == 'resnet_pretrained_freeze':
            from feature_extractors.resnet import resnet18
            model = resnet18(True, dimension=input_dimension[1])
            model.freeze_layers([model.conv1, model.bn1, model.relu,
                                 model.maxpool, model.layer1, model.layer2, model.layer3])
            return model

        else:
            raise Exception()
    else:
        return pretrained_feature_extractor
