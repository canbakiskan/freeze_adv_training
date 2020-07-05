import torch

from freeze_AT.src.utils.get_modules import (
    get_classifier,
)
from freeze_AT.src.utils.parameters import get_arguments

from freeze_AT.src.utils.namers import classifier_ckpt_namer


def check_match(model, reference_model, args):
    is_matching = True
    if args.AT_first:
        for p, ref_p in zip(model.named_parameters(), reference_model.named_parameters()):
            parameter_name = p[0]
            is_parameter_AT = False
            for layer_name in args.adv_training_layers:
                if layer_name in parameter_name:
                    is_parameter_AT |= True
            if is_parameter_AT:
                is_matching &= torch.all(p[1] == ref_p[1]).item()
            else:
                is_matching &= torch.all(p[1] != ref_p[1]).item()

            if not is_matching:
                print("==================")
                print(f"Model: {classifier_ckpt_namer(args)}")
                print(f"{p[0]} is not matching")
                break

    elif args.NT_first:
        for p, ref_p in zip(model.named_parameters(), reference_model.named_parameters()):
            parameter_name = p[0]
            is_parameter_AT = False
            for layer_name in args.adv_training_layers:
                if layer_name in parameter_name:
                    is_parameter_AT |= True
            if not is_parameter_AT:
                is_matching &= torch.all(p[1] == ref_p[1]).item()
            else:
                is_matching &= torch.all(p[1] != ref_p[1]).item()
            if not is_matching:
                print("==================")
                print(f"Model: {classifier_ckpt_namer(args)}")
                print(f"{p[0]} is not matching")
                break

    return is_matching


args = get_arguments()

args.AT_first = False
args.NT_first = False

args.adv_training_layers = ["all"]
AT_model = get_classifier(args)

args.adv_training_layers = ["none"]
NT_model = get_classifier(args)

AT_layers = ["init_conv", "block1", "block2", "block3", "last_bn", "linear"]

for i in range(9):
    if i < 4:
        current_AT_layers = AT_layers[i+1:]
    elif i > 4:
        current_AT_layers = AT_layers[:i-4]
    else:
        continue
    args.adv_training_layers = current_AT_layers

    args.AT_first = True
    args.NT_first = False
    try:
        model = get_classifier(args)
        check_match(model, AT_model, args)
    except:
        continue

    args.AT_first = False
    args.NT_first = True
    try:
        model = get_classifier(args)
        check_match(model, NT_model, args)
    except:
        continue
