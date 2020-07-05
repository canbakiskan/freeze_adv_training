# !/bin/bash 

pid=81154

while [ -d /proc/$pid ] ; do
    sleep 1
done

# COMMAND="python train_classifier.py --adv_training_layers init_conv --NT_first"
# echo $COMMAND
# eval $COMMAND


# COMMAND="python train_classifier.py --adv_training_layers init_conv --AT_first"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python train_classifier.py --adv_training_layers init_conv block1 --NT_first"
# echo $COMMAND
# eval $COMMAND


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
