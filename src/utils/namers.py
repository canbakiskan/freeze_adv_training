import torch
import os
import numpy as np


def adv_training_params_string(args):
    adv_training_params_string = ""
    if args.adv_training_attack:
        adv_training_params_string += f"_{args.adv_training_attack}"
        adv_training_params_string += (
            f"_eps_{np.int(np.round(args.adv_training_epsilon*255))}"
        )
        if "EOT" in args.adv_training_attack:
            adv_training_params_string += f"_Ne_{args.adv_training_EOT_size}"
        if "PGD" in args.adv_training_attack or "CW" in args.adv_training_attack:
            adv_training_params_string += f"_Ns_{args.adv_training_num_steps}"
            adv_training_params_string += (
                f"_ss_{np.int(np.round(args.adv_training_step_size*255))}"
            )
            adv_training_params_string += f"_Nr_{args.adv_training_num_restarts}"
        if "FGSM" in args.adv_training_attack:
            adv_training_params_string += (
                f"_a_{np.int(np.round(args.adv_training_alpha*255))}"
            )

        if args.AT_first:
            adv_training_params_string += "_AT_first"
        elif args.NT_first:
            adv_training_params_string += "_NT_first"

        # if args.adv_training_layers[0] != "none":
        adv_training_params_string += "_ATlayers"
        for layer in args.adv_training_layers:
            adv_training_params_string += f"_{layer}"

        if args.adv_training_third_time:
            adv_training_params_string += f"_thirdtime"

    return adv_training_params_string


def classifier_params_string(args):
    classifier_params_string = args.classifier_arch

    classifier_params_string += adv_training_params_string(args)

    if not args.adv_training_attack:
        classifier_params_string += "_NT"

    classifier_params_string += f"_ep_{args.epochs}"

    classifier_params_string += f"_{args.lr_scheduler}"

    return classifier_params_string


def attack_params_string(args):
    attack_params_string = f"{args.attack_box_type}"
    if args.attack_box_type == "other":
        attack_params_string += f"_{args.attack_otherbox_type}"
        attack_params_string += f"_eps_{np.int(np.round(args.attack_epsilon*255))}"

    elif args.attack_box_type == "white":
        attack_params_string += f"_{args.attack_method}"
        attack_params_string += f"_eps_{np.int(np.round(args.attack_epsilon*255))}"
        if "EOT" in args.attack_method:
            attack_params_string += f"_Ne_{args.attack_EOT_size}"
        if "PGD" in args.attack_method or "CW" in args.attack_method:
            attack_params_string += f"_Ns_{args.attack_num_steps}"
            attack_params_string += f"_ss_{np.int(np.round(args.attack_step_size*255))}"
            attack_params_string += f"_Nr_{args.attack_num_restarts}"
        if "RFGSM" in args.attack_method:
            attack_params_string += f"_a_{np.int(np.round(args.attack_alpha*255))}"

    return attack_params_string


def classifier_ckpt_namer(args):

    file_path = args.directory + f"checkpoints/classifiers/"

    file_path += classifier_params_string(args)

    file_path += ".pt"

    return file_path


def classifier_log_namer(args):

    file_path = args.directory + f"logs/"

    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path


def attack_file_namer(args):

    file_path = args.directory + f"data/attacked_dataset/"

    file_path += attack_params_string(args)
    file_path += "_"
    file_path += classifier_params_string(args)

    file_path += ".npy"

    return file_path


def attack_log_namer(args):

    file_path = args.directory + f"logs/"

    file_path += attack_params_string(args)
    file_path += "_"
    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path
