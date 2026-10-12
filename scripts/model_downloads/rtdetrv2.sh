# RT-DETRv2 R50-vd COCO weights in the Transformers format.
# Published by PekingU, pinned to one repository revision.
base_url='https://huggingface.co/PekingU/rtdetr_v2_r50vd/resolve/282494075698cab9faa1096ae26856890030c817'

download rtdetrv2/config.json "$base_url/config.json" \
    'd754c04c9a4c6d015d8393fa2dae4c63f4a9f0ba173c3d1a66e55cff59d8f060'

download rtdetrv2/preprocessor_config.json "$base_url/preprocessor_config.json" \
    'cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1'

download rtdetrv2/model.safetensors "$base_url/model.safetensors" \
    '3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71'
