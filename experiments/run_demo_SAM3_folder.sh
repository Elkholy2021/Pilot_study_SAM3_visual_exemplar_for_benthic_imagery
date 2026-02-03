


N=10

for image in $(ls /home/hassan/Downloads/2A_media2/downloaded_images/test_sam3_concept/*.jpg | head -n $N); do
    python scripts/run_demo_SAM3.py \
        --model /media/hassan/Data_vault1/datasets_hassan/models/sam3.pt \
        --image_path "$image" \
        --THR 0.5
done
