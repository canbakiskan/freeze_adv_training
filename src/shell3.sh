# !/bin/bash 

pid=220768

while [ -d /proc/$pid ] ; do
    sleep 1
done


# COMMAND="python train_classifier.py --adv_training_layers block1 block2 block3 last_bn linear --NT_first"
# echo $COMMAND
# eval $COMMAND


# COMMAND="python train_classifier.py --adv_training_layers block1 block2 block3 last_bn linear --AT_first"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python train_classifier.py --adv_training_layers block2 block3 last_bn linear --NT_first"
# echo $COMMAND
# eval $COMMAND

COMMAND="python train_classifier.py --adv_training_layers init_conv block1 block2 block3 last_bn linear --NT_first"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --adv_training_layers init_conv block1 --AT_first"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --adv_training_layers init_conv block1 block2 --NT_first"
echo $COMMAND
eval $COMMAND


COMMAND="python train_classifier.py --adv_training_layers init_conv block1 block2 --AT_first"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --adv_training_layers init_conv block1 block2 block3 --NT_first"
echo $COMMAND
eval $COMMAND


COMMAND="python train_classifier.py --adv_training_layers init_conv block1 block2 block3 --AT_first"
echo $COMMAND
eval $COMMAND


COMMAND="python train_classifier.py --adv_training_layers block2 block3 last_bn linear --AT_first"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --adv_training_layers block3 last_bn linear --NT_first"
echo $COMMAND
eval $COMMAND


COMMAND="python train_classifier.py --adv_training_layers block3 last_bn linear --AT_first"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --adv_training_layers last_bn linear --NT_first"
echo $COMMAND
eval $COMMAND


COMMAND="python train_classifier.py --adv_training_layers last_bn linear --AT_first"
echo $COMMAND
eval $COMMAND
