export PYTHONPATH=/path/to/your/project/root:$PYTHONPATH

echo "Start generating dataset..."

python \
    data_tools/senna_nusc_data_converter_api.py \
    nuscenes \
    --root-path /storage/data-acc/nuscenes \
    --out-dir ./data \
    --extra-tag senna_nusc \
    --version v1.0-mini \
    --canbus /storage/data-acc/nuscenes/ \
    --num-workers 32