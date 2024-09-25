python3 llava/eval/run_llava.py --image-file "EE-LLaVA/test_images/IMG_8458 copy.jpg" --query "what is the aircraft?" --model-path llava-v1.6-vicuna-7b/





python model_vqa.py \
    --model-path ./checkpoints/llava-v1.6-vicuna-7b \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    playground/data/coco2014_val/out_coco \
    --answers-file \
    EE-LLaVA/qa90_coco_llava_test.jsonl



python model_vqa.py \
    --model-path ./checkpoints/llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    playground/data/coco2014_val/out_coco \
    --answers-file \
    EE-LLaVA/qa90_coco_llava_test_2.jsonl


python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    EE-LLaVA/qa90_coco_llava_test_2.jsonl \
    EE-LLaVA/qa90_coco_llava_test_2.jsonl \
    --rule llava/eval/table/rule.json \
    --output EE-LLaVA/review_2.json

python summarize_gpt_review.py \
    --dir EE-LLaVA \
    --files review_2.json

# aim 

python model_vqa.py \
    --model-path ./checkpoints/llava-v1.5-7b-lora_aim_singlenode \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    playground/data/coco2014_val/out_coco \
    --answers-file \
    EE-LLaVA/qa90_coco_llava_aim.jsonl


python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    EE-LLaVA/qa90_coco_llava_aim.jsonl \
    EE-LLaVA/qa90_coco_llava_test_3.jsonl \
    --rule llava/eval/table/rule.json \
    --output EE-LLaVA/review_aim.json

python summarize_gpt_review.py \
    --dir EE-LLaVA \
    --files review_aim.json