#!/bin/bash -e 


# declare -a TASKS=("tracking_shuffled_objects_seven_objects" "salient_translation_error_detection" "tracking_shuffled_objects_three_objects" "geometric_shapes" "object_counting" "word_sorting" "logical_deduction_five_objects" "hyperbaton" "sports_understanding" "logical_deduction_seven_objects" "multistep_arithmetic_two" "ruin_names" "causal_judgement" "logical_deduction_three_objects" "formal_fallacies" "snarks" "boolean_expressions" "reasoning_about_colored_objects" "dyck_languages" "navigate" "disambiguation_qa" "temporal_sequences" "web_of_lies" "tracking_shuffled_objects_five_objects" "penguins_in_a_table" "movie_recommendation" "date_understanding")
declare -a TASKS=("dyck_languages")


for task in "${TASKS[@]}"
do
    nohup python3 -u run_eval_bbh.py $task paper > 2aug_nohup_${task}_tutorial.log 2>&1  &
    # python3 -u run_eval_bbh.py $task tutorial 2>&1 > log_${task}_tutorial.log;
    echo "ran task $task"
done