# uv run packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py \
#         --checkpoint-path /home/emon/Models/LTX-2/checkpoints/ltx-2-19b-dev-fp8.safetensors \
#         --gemma-root /home/emon/Models/LTX-2/checkpoints/gemma-3-12b \
#         --enhance-prompt \
#         --prompt "Animate this scene" \
#         --image /home/emon/Models/LTX-2/inputs/live-inputs/IMG_7550.JPG 0 1 \
#         --output-path /home/emon/Models/LTX-2/outputs/live-video-image-1.mp4 \
#         --enable-fp8 > run.log 2>&1
        # --distilled-lora /home/emon/Models/LTX-2/checkpoints/ltx-2-19b-distilled-lora-384.safetensors \
        # --spatial-upsampler-path /home/emon/Models/LTX-2/checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors \

# uv run packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py \
#         --checkpoint-path /home/emon/Models/LTX-2/checkpoints/ltx-2-19b-distilled-fp8.safetensors \
#         --gemma-root /home/emon/Models/LTX-2/checkpoints/gemma-3-12b \
#         --enhance-prompt \
#         --prompt "Animate this scene" \
#         --image /home/emon/Models/LTX-2/inputs/live-inputs/IMG_7550.JPG 0 0.5 \
#         --output-path /home/emon/Models/LTX-2/outputs/live-video-two-stages-image-0.5.mp4 \
#         --enable-fp8 > run.log 2>&1

# checkpoint_path="/home/emon/Models/LTX-2/checkpoints/ltx-2-19b-distilled-lora-384.safetensors"
checkpoint_path="/home/emon/Models/LTX-2/checkpoints/ltx-2-19b-dev-fp8.safetensors"
gemma_root="/home/emon/Models/LTX-2/checkpoints/gemma-3-12b"
spatial_upsampler_path="/home/emon/Models/LTX-2/checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors"
lora_path="/home/emon/Models/LTX-2/checkpoints/ltx-2-19b-ic-lora-pose-control.safetensors"
video_reference="/home/emon/Models/LTX-2/inputs/videos/1769592758264.MP4"

input_folder=$1
prompt=$2
output_folder="outputs/ic-lora/viral-dance/run_$(date +'%Y-%m-%d_%H-%M-%S')"

frame_rate=24
video_duration=10
num_frames=$(($frame_rate * $video_duration))
# output must have 1 + 8 * x frames (e.g., 1, 9, 17, ...)
num_frames=$(($num_frames - $num_frames % 8 + 1))

min_side=512

if [ ! -d "$output_folder" ]; then
        mkdir -p "$output_folder"
fi

for file in $input_folder/*; do
        filename=$(basename "$file")
        filename="${filename%.*}"

        output_file="$filename.mp4"
        output_log="$filename.log"

        input_width=$(identify -format "%w" "$file")
        input_height=$(identify -format "%h" "$file")

        # min side 768 and other side divisible by 64
        if [ $input_width -lt $input_height ]; then
                output_height=$((input_height * $min_side / input_width))
                output_height=$((output_height - output_height % 64))
                output_width=$min_side
        else
                output_width=$((input_width * $min_side / input_height))
                input_width=$((input_width - input_width % 64))
                output_height=$min_side
        fi

        echo "==========================================="
        echo -e "Processing $file\n"
        echo -e "Input width\t: $input_width"
        echo -e "Input height\t: $input_height"
        echo -e "Output width\t: $output_width"
        echo -e "Output height\t: $output_height"
        echo -e "Output file\t: $output_file"
        echo -e "Output log\t: $output_log"
        echo "==========================================="
        uv run packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py \
                --checkpoint-path $checkpoint_path \
                --gemma-root $gemma_root \
                --spatial-upsampler-path $spatial_upsampler_path \
                --video-conditioning $video_reference 0.2 \
                --enhance-prompt \
                --enable-fp8 \
                --prompt "$prompt" \
                --width $output_width \
                --height $output_height \
                --image $file 0 0.8 \
                --output-path $output_folder/$output_file \
                --lora $lora_path 1 \
                --seed 8765876575 \
                --num-frames $num_frames \
                > $output_folder/$output_log 2>&1
done
                # --frame-rate $frame_rate \