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
    EE-LLaVA/qa90_coco_llava_test.jsonl