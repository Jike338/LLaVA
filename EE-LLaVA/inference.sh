python3 llava/eval/run_llava.py --image-file "EE-LLaVA/test_images/IMG_8458 copy.jpg" --query "what is the aircraft?" --model-path llava-v1.6-vicuna-7b/





python model_vqa.py \
    --model-path ./checkpoints/llava-v1.5-7b-lora \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    playground/data/coco2014_val/out_coco \
    --answers-file \
    playground/data/coco2014_val_qa_eval/qa90_questions_llava_test.jsonl