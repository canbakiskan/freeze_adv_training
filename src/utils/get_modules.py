import numpy as np
import torch
from os import path
from freeze_AT.src.utils.namers import classifier_ckpt_namer


def get_classifier(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.classifier_arch == "resnetwide":
        from freeze_AT.src.models.resnet import ResNetWide

        classifier = ResNetWide().to(device)

    elif args.classifier_arch == "resnet":
        from freeze_AT.src.models.resnet import ResNet

        classifier = ResNet().to(device)

    else:
        raise NotImplementedError

    classifier.load_state_dict(
        torch.load(classifier_ckpt_namer(args),
                   map_location=torch.device(device),)
    )
    print(f"Classifier: {classifier_ckpt_namer(args)}")

    return classifier
