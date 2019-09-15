## PROJECT INSTALLATION

### Create a venv
	virtualenv -p python3 <venv_dir>
	source <venv_dir>/bin/activate
	
### Install the requirements
    #from the <project_dir>
    pip install -r requirements.txt

### Libs Installation
see more at: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

    - COCO API installation
        git clone https://`github.com/cocodataset/cocoapi.git
        cd cocoapi/PythonAPI
        make
        cp -r pycocotools <path_to_tensorflow>/models/research/
        
    - protobuf-compiler installation (depend on the OS)
        
    - Protobuf Compilation
        # From tensorflow/models/research/
        protoc object_detection/protos/*.proto --python_out=.
        
    - Add Libraries to PYTHONPATH
        # From tensorflow/models/research/
        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
        
    - Testing the Installation
        # From tensorflow/models/research/
        python object_detection/builders/model_builder_test.py

## CREATION OF MAPPY RECORDS

    python <project_dir>/models/research/object_detection/dataset_tools/create_mappy_blur_tf_record.py
    --label_map_path="<project_dir>/models/research/object_detection/data/mappy_blur_label_map.pbtxt" --data_dir="<project_dir>/mappy_annotations" --output_dir="<project_dir>/mappy_proto_records"

## SETTING UP THE PROJECT ON GOOGLE CLOUD

    + Create a GCP project and Enable the ML Engine APIs. (https://cloud.google.com/resource-manager/docs/creating-managing-projects, https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component&_ga=1.73374291.1570145678.1496689256)
        Nom du projet
            mappy blur ia
        ID du projet
            mappy-blur-ia
    
        Ressource > Storage > Buckets
            mappy_blur_ia_tf_model_bucket : Nearline	europe-north1 (Finlande)	Région	Par objet	Indisponible    Stratégie du bucket et LCA
        
    + install the Google Cloud SDK
        https://cloud.google.com/sdk/install
        
        gcloud init
        
    + Set up a Google Cloud Storage (GCS) bucket. (https://cloud.google.com/storage/docs/creating-buckets)
        #Substitute ${YOUR_GCS_BUCKET} with the name of your bucket
        export YOUR_GCS_BUCKET=${YOUR_GCS_BUCKET}
        
    + Push the mappy records on to the ressource of the GCP project
        gsutil cp <project_dir>/mappy_proto_records/mappy_blur_train.record-* gs://${YOUR_GCS_BUCKET}/data/
        gsutil cp <project_dir>/mappy_proto_records/mappy_blur_val.record-* gs://${YOUR_GCS_BUCKET}/data/
    
    + Push the map file
        gsutil cp <project_dir>/models/research/object_detection/data/mappy_blur_label_map.pbtxt gs://${YOUR_GCS_BUCKET}/data/mappy_blur_label_map.pbtxt
        
    + Push the Object Detection Pipeline Configuration
        gsutil cp <project_dir>/models/research/object_detection/samples/configs/faster_rcnn_resnet101_mappy_blur.config gs://${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101_mappy_blur.config

## PUSH THE faster_rcnn_resnet101 PRE-TRAINED MODEL
    wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    gsutil cp <dir>/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/

##CHECKING GOOGLE CLOUD STRORAGE BUCKET 
see more at: https://console.cloud.google.com/storage/browser?project=mappy-blur-ia&folder&organizationId=489668500446
+ ${YOUR_GCS_BUCKET}/
  + data/
    - faster_rcnn_resnet101_mappy_blur.config
    - mappy_blur_label_map.pbtxt
    - mappy_blur_train.record--*
    - mappy_blur_val.record-*
    - model.ckpt.data-00000-of-00001
    - model.ckpt.index
    - model.ckpt.meta

## TRAINING AND EVALUATION JOBS ON GOOGLE CLOUD ML ENGINE

    + package the Tensorflow Object Detection code
               
        # run the following commands from the <project_dir>/models/research/ directory:
        bash <project_dir>/models/research/object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
        python setup.py sdist
        (cd slim && python setup.py sdist)
        
        This will create python packages dist/object_detection-0.1.tar.gz, slim/dist/slim-0.1.tar.gz, and /tmp/pycocotools/pycocotools-2.0.tar.gz.
        
        The configuration file can be found at object_detection/samples/cloud/cloud.yml
        
    + To start training and evaluation
        
        # From <project_dir>/models/research/
        gcloud ml-engine jobs submit training 'whoami'_object_detection_mappy_blur_'date +%m_%d_%Y_%H_%M_%S'
         --runtime-version 1.12
         --job-dir=gs://${YOUR_GCS_BUCKET}/model_dir
         --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz
         --module-name object_detection.model_main
         --region europe-west1
         --config object_detection/samples/cloud/cloud.yml
         --
         --model_dir=gs://${YOUR_GCS_BUCKET}/model_dir
         --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101_mappy_blur.config

## MONITORING PROGRESSS WITH TENSORBOARD
You can monitor progress of the training and eval jobs by running Tensorboard on your local machine:

    # This command needs to be run once to allow your local machine to access your
    # GCS bucket.
    gcloud auth application-default login
        
    tensorboard --logdir=gs://${YOUR_GCS_BUCKET}/model_dir

## EXPORTING THE TENSORFLOW GRAPH
    # From <project_dir>/models/research/
    gsutil cp gs://${YOUR_GCS_BUCKET}/model_dir/model.ckpt-${CHECKPOINT_NUMBER}.* <project_dir>/mappy_trained_models
    python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path object_detection/samples/configs/faster_rcnn_resnet101_mappy_blur.config \
        --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
        --output_directory <project_dir>/mappy_trained_models/exported_graphs


gsutil cp gs://${YOUR_GCS_BUCKET}/model_dir/model.ckpt-6055.* .

python object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path object_detection/samples/configs/faster_rcnn_resnet101_mappy_blur.config \
--trained_checkpoint_prefix model.ckpt-6055 \
--output_directory exported_graphs
