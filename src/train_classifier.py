import time
import os
from tqdm import tqdm
import numpy as np

import logging

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from freeze_AT.src.models.resnet import ResNet, ResNetWide


from freeze_AT.src.utils.parameters import get_arguments
from freeze_AT.src.utils.read_datasets import cifar10

from freeze_AT.src.utils.namers import (
    classifier_ckpt_namer,
    classifier_log_namer,
)
from freeze_AT.src.utils.get_modules import (
    get_classifier,
)

from freeze_AT.adversarial_framework.torchattacks import (
    PGD,
    FGSM,
    RFGSM
)
from freeze_AT.src.train_test_functions import (
    train_epoch,
    test_epoch,
)
logger = logging.getLogger(__name__)


def weights_init(m):
    if type(m[1]) in {torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d}:
        m[1].reset_parameters()
        if isinstance(m[1], torch.nn.BatchNorm2d):
            m[1].reset_running_stats()


def adjust_req_grad_init_weights(model, args, initialize_weights=True, print_values=False):

    AT_or_NT = "AT" if args.NT_first else "NT"

    for p in model.parameters():
        p.requires_grad = False

    if print_values:
        for p in model.named_parameters():
            try:
                print(p[0], p[1].requires_grad, p[1][0, 0, 0, 0])
            except:
                print(p[0], p[1].requires_grad, p[1][0])

    for m in model.named_modules():
        if type(m[1]) in {torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d}:
            parameter_name = m[0]
            is_parameter_AT = False
            for layer_name in args.adv_training_layers:
                if layer_name in parameter_name:
                    is_parameter_AT |= True

            requires_grad = is_parameter_AT if AT_or_NT == "AT" else not is_parameter_AT
            if isinstance(m[1].weight, torch.Tensor):
                m[1].weight.requires_grad = requires_grad
            if isinstance(m[1].bias, torch.Tensor):
                m[1].bias.requires_grad = requires_grad
            if requires_grad and initialize_weights:
                weights_init(m)

    if print_values:
        for p in model.named_parameters():
            try:
                print(p[0], p[1].requires_grad, p[1][0, 0, 0, 0])
            except:
                print(p[0], p[1].requires_grad, p[1][0])


def get_frozen_model(args):

    if args.adv_training_third_time:
        args.adv_training_third_time = False
        model = get_classifier(args)
        args.adv_training_third_time = True

    else:
        AT_layers_save = args.adv_training_layers
        AT_first_save = args.AT_first
        NT_first_save = args.NT_first
        args.adv_training_layers = ["all"] if args.AT_first else ["none"]
        args.AT_first = False
        args.NT_first = False
        model = get_classifier(args)
        args.adv_training_layers = AT_layers_save
        args.AT_first = AT_first_save
        args.NT_first = NT_first_save

    return model


def get_optimizer_scheduler(model,  args, train_loader_length):

    if args.lr_scheduler == "cyclic":
        # Which optimizer to be used for training
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        lr_steps = args.epochs * train_loader_length
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_min,
            max_lr=args.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    else:
        from torch.optim.lr_scheduler import MultiStepLR

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

        scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    return (optimizer, scheduler)


def save_checkpoint(model, args):
    if not os.path.exists(args.directory + "checkpoints/classifiers/"):
        os.makedirs(args.directory + "checkpoints/classifiers/")

    classifier_filepath = classifier_ckpt_namer(args)
    torch.save(
        model.state_dict(), classifier_filepath,
    )

    logger.info(f"Saved to {classifier_filepath}")


def train(model, data_loaders, optimization, args):

    (train_loader, test_loader) = data_loaders
    (optimizer, scheduler) = optimization

    data_params = {"x_min": 0.0, "x_max": 1.0}

    train_args = dict(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    attacks = dict(
        PGD=PGD,
        FGSM=FGSM,
        RFGSM=RFGSM,
    )

    adversarial_test_params = {
        "norm": "inf",
        "eps": args.attack_epsilon,
        "alpha": args.attack_alpha,
        "step_size": args.attack_step_size,
        "num_steps": args.attack_num_steps,
        "random_start": (
            args.attack_rand and args.attack_num_restarts > 1
        ),
        "num_restarts": args.attack_num_restarts,
    }

    if "CWlinf" in args.attack_method:
        attack_method = args.attack_method.replace(
            "CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        attack_method = args.attack_method
        loss_function = "cross_entropy"

    adversarial_test_args = dict(
        attack=attacks[attack_method],
        attack_args=dict(
            net=model, data_params=data_params, attack_params=adversarial_test_params
        ),
        loss_function=loss_function
    )

    test_args = dict(model=model, test_loader=test_loader)

    if args.NT_first:  # means we need to do AT training

        adversarial_train_params = {
            "norm": "inf",
            "eps": args.adv_training_epsilon,
            "alpha": args.adv_training_alpha,
            "step_size": args.adv_training_step_size,
            "num_steps": args.adv_training_num_steps,
            "random_start": (
                args.adv_training_rand and args.adv_training_num_restarts > 1
            ),
            "num_restarts": args.adv_training_num_restarts,
        }

        if "CWlinf" in args.adv_training_attack:
            adv_training_attack = args.adv_training_attack.replace(
                "CWlinf", "PGD")
            loss_function = "carlini_wagner"
        else:
            adv_training_attack = args.adv_training_attack
            loss_function = "cross_entropy"

        adversarial_train_args = dict(
            attack=attacks[adv_training_attack],
            attack_args=dict(
                net=model, data_params=data_params, attack_params=adversarial_train_params
            ),
            loss_function=loss_function,
            AT_layers=args.adv_training_layers
        )

        train_args["adversarial_args"] = adversarial_train_args

        logger.info(args.adv_training_attack + " training")

    else:
        logger.info("Natural training")

    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

    test_loss, test_acc = test_epoch(**test_args)
    test_loss_adv, test_acc_adv = test_epoch(
        **test_args, adversarial_args=adversarial_test_args)

    logger.info(
        f"Initial Test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")
    logger.info(
        f"Initial Adv \t loss: {test_loss_adv:.4f} \t acc: {test_acc_adv:.4f}")

    for epoch in tqdm(range(1, args.epochs + 1)):
        start_time = time.time()

        train_loss, train_acc = train_epoch(**train_args)

        test_loss, test_acc = test_epoch(**test_args)
        test_loss_adv, test_acc_adv = test_epoch(
            **test_args, adversarial_args=adversarial_test_args)

        end_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info(
            f"{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}"
        )
        logger.info(
            f"Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")
        logger.info(
            f"Adv  \t loss: {test_loss_adv:.4f} \t acc: {test_acc_adv:.4f}")


def setup():

    args = get_arguments()

    assert args.NT_first ^ args.AT_first
    assert args.adv_training_layers[0] != "none" and args.adv_training_layers[0] != "all"

    if not os.path.exists(args.directory + "logs"):
        os.mkdir(args.directory + "logs")

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(classifier_log_namer(args)),
            logging.StreamHandler(),
        ],
    )

    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


def main():
    """ main function to run the experiments """

    args, device = setup()
    data_loaders = cifar10(args)

    model = get_frozen_model(args)
    model.train()

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimization_tuple = get_optimizer_scheduler(
        model, args, len(data_loaders[0]))

    if args.adv_training_third_time:
        args.NT_first = not args.NT_first
        args.AT_first = not args.AT_first

        adjust_req_grad_init_weights(
            model, args, initialize_weights=False)
        train(model, loaders, optimization_tuple, args)

        args.NT_first = not args.NT_first
        args.AT_first = not args.AT_first

    else:
        adjust_req_grad_init_weights(model, args)
        train(model, data_loaders, optimization_tuple, args)

    if args.save_checkpoint:
        save_checkpoint(model, args)


if __name__ == "__main__":
    main()
