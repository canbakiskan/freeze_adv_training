from tqdm import tqdm

from freeze_AT.src.utils.namers import (
    attack_log_namer,
    attack_file_namer,
)
from freeze_AT.src.utils.get_modules import (
    get_classifier,
)

import numpy as np
import torch
import torch.nn.functional as F
from freeze_AT.adversarial_framework.torchattacks import (
    PGD,
    FGSM,
    RFGSM,
)
from freeze_AT.src.utils.read_datasets import cifar10, cifar10_blackbox
from freeze_AT.adversarial_framework.torchdefenses import adversarial_test

import logging

logger = logging.getLogger(__name__)


def generate_attack(args, model, data, target, adversarial_args):

    if args.attack_box_type == "white":

        if args.attack_whitebox_type == "SW":
            adversarial_args["attack_args"]["net"] = model.module_outer
            adversarial_args["attack_args"]["attack_params"]["EOT_size"] = 1

        else:
            adversarial_args["attack_args"]["net"] = model

    elif args.attack_box_type == "other":
        if "-T" in args.attack_otherbox_type:
            # it shouldn't enter this clause
            raise ValueError

        elif args.attack_otherbox_type == "decision":
            # this attack fails to find perturbation for misclassification in its
            # initialization part. Then it quits.
            import foolbox as fb

            fmodel = fb.PyTorchModel(model, bounds=(0, 1))

            attack = fb.attacks.BoundaryAttack(
                init_attack=fb.attacks.LinearSearchBlendedUniformNoiseAttack(
                    # directions=100000, steps=1000,
                ),
                # init_attack=fb.attacks.LinfDeepFoolAttack(steps=100),
                steps=2500,
                spherical_step=0.01,
                source_step=0.01,
                source_step_convergance=1e-07,
                step_adaptation=1.5,
                tensorboard=False,
                update_stats_every_k=10,
            )
            # attack = fb.attacks.BoundaryAttack()
            epsilons = [8 / 255]
            _, perturbation, success = attack(
                fmodel, data, target, epsilons=epsilons)
            return perturbation[0] - data

        else:
            raise ValueError

    adversarial_args["attack_args"]["x"] = data
    adversarial_args["attack_args"]["y_true"] = target
    perturbation = adversarial_args["attack"](
        **adversarial_args["attack_args"])

    return perturbation


def main():

    from freeze_AT.src.utils.parameters import get_arguments

    args = get_arguments()

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(attack_log_namer(args)),
            logging.StreamHandler(),
        ],
    )
    logger.info(args)
    logger.info("\n")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.attack_box_type == "other" and "-T" in args.attack_otherbox_type:
        args.save_attack = False

    classifier = get_classifier(args)
    model = classifier
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    _, test_loader = cifar10(args)
    test_loss, test_acc = adversarial_test(model, test_loader)
    logger.info(f"Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    attacks = dict(
        PGD=PGD,
        FGSM=FGSM,
        RFGSM=RFGSM,
    )

    attack_params = {
        "norm": "inf",
        "eps": args.attack_epsilon,
        "alpha": args.attack_alpha,
        "step_size": args.attack_step_size,
        "num_steps": args.attack_num_steps,
        "random_start": (args.adv_training_rand and args.adv_training_num_restarts > 1),
        "num_restarts": args.attack_num_restarts
    }

    data_params = {"x_min": 0.0, "x_max": 1.0}

    if "CWlinf" in args.attack_method:
        attack_method = args.attack_method.replace("CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        attack_method = args.attack_method
        loss_function = "cross_entropy"

    adversarial_args = dict(
        attack=attacks[attack_method],
        attack_args=dict(
            net=model,
            data_params=data_params,
            attack_params=attack_params,
            progress_bar=args.attack_progress_bar,
            verbose=True,
            loss_function=loss_function,
        ),
    )
    test_loss = 0
    correct = 0

    if args.save_attack:
        attacked_images = torch.zeros(10000, 3, 32, 32)

    output_attack = torch.zeros(10000, 10)

    if args.attack_box_type == "other" and "-T" in args.attack_otherbox_type:
        test_loader = cifar10_blackbox(args)
    else:
        _, test_loader = cifar10(args)

    for batch_idx, (data, target) in enumerate(
        tqdm(test_loader, desc="Attack progress", leave=False)
    ):

        data = data.to(device)
        target = target.to(device)

        if not (args.attack_box_type == "other" and "-T" in args.attack_otherbox_type):
            attack_batch = generate_attack(
                args, model, data, target, adversarial_args)
            data += attack_batch
            data = data.clamp(0.0, 1.0)
            if args.save_attack:
                attacked_images[
                    batch_idx
                    * args.test_batch_size: (batch_idx + 1)
                    * args.test_batch_size,
                ] = data.detach().cpu()

        with torch.no_grad():
            output_attack[
                batch_idx
                * args.test_batch_size: (batch_idx + 1)
                * args.test_batch_size,
            ] = (model(data).detach().cpu())

    _, test_loader = cifar10(args)
    target = torch.tensor(test_loader.dataset.targets)
    pred_attack = output_attack.argmax(dim=1, keepdim=True)

    correct_attack = pred_attack.eq(target.view_as(pred_attack)).sum().item()
    accuracy_attack = correct_attack / 10000

    logger.info(f"Attack accuracy: {(100*accuracy_attack):.2f}%")

    if args.save_attack:
        attack_filepath = attack_file_namer(args)
        np.save(attack_filepath, attacked_images.detach().cpu().numpy())

        logger.info(f"Saved to {attack_filepath}")


if __name__ == "__main__":
    main()
